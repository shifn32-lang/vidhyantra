import datetime
import json
import logging
import queue
import re
import threading
import time
from zoneinfo import ZoneInfo

from django.conf import settings
from openai import OpenAI

from myapp import business_info

logger = logging.getLogger(__name__)
NVIDIA_CHAT_MODEL = getattr(settings, 'NVIDIA_CHAT_MODEL', 'nvidia/nemotron-3.5-lightning-30b-a3b')

MAX_TOKENS = 2048
TEMPERATURE = 0.5          # baseline/fallback
TEMPERATURE_PRECISE = 0.4      # maths or code turns — consistency matters more than variety here
TEMPERATURE_CONVERSATIONAL = 0.7  # everyday chat turns — a flat low temperature made phrasing repetitive/stiff
TOP_P = 0.95
# Two retries instead of one — real reports (AIReport #10, #30) showed a
# single retry giving up too early on a merely-busy worker. Each retry only
# costs time on the already-broken failure path (a successful first attempt
# is completely unaffected), so the extra patience is close to free in
# exchange for turning more transient blips into a real answer instead of
# "did not respond".
STREAM_RETRY_ATTEMPTS = 2
STREAM_RETRY_BACKOFF_SECONDS = 0.5
STREAM_TIMEOUT_DEFAULT = 15.0
# A longer request budget for turns that were deliberately given a larger
# max_tokens (coding, file generation, explicit "complete"/"detailed"
# requests — see wants_long_form_output below) — the 25s default is tuned
# for an ordinary chat reply, not a multi-thousand-token document.
STREAM_TIMEOUT_LONG = 60.0

# EduTrellis Vision was live-tested to randomly (~1 in 3 tries, reproducible
# across many prompt-wording variants and even at temperature 0) open with a
# flat denial that any image was attached, despite one actually being there
# — an unreliability in the underlying vision model itself, not something
# prompt-wording alone fixes. This catches that specific opening so
# stream_chat can silently retry with a fresh generation before anything
# reaches the user, instead of them seeing a wrong "I can't see an image".
#
# Deliberately broad — the model phrases this denial many different ways
# ("please upload the image", "I don't have the capability", "as an AI...",
# "there might be a misunderstanding" — see git history for the ~17 real
# variants this was validated against with zero false positives against
# real successful descriptions), and missing a variant is far worse
# (a wrong answer reaches the user) than an occasional unnecessary retry.
_VISION_NO_IMAGE_RE = re.compile(
    r"(I(?:'m| am) unable to|"
    r"I can'?t (?:\w+ )?(?:see|view|read|analyze|access|assist)|"
    r"I cannot (?:\w+ )?(?:see|view|read|analyze|access)|"
    r"I don'?t have (?:the |any )?(?:capability|ability|access)|"
    r"I do not have (?:the |any )?(?:capability|ability|access)|"
    r"as an AI[, ]|"
    r"please (?:upload|provide|share|describe)\b[^.!?]{0,25}\bimage|"
    r"(?:have not|haven'?t) provided (?:an |the )?image|"
    r"(?:no|don'?t see any|do not see any) image (?:was )?(?:attached|provided|uploaded)?|"
    r"not (?:possible|capable)[^.!?]{0,40}text-?based|"
    r"rely on textual descriptions?|"
    r"there (?:might|may) (?:be|have been) (?:some )?(?:confusion|misunderstanding))",
    re.IGNORECASE,
)
# How much of the reply to hold back before deciding it's clean — long
# enough that every observed failure phrasing shows up well within it (one
# live-tested case buried its denial after ~190 characters of preamble, past
# the original 200-char threshold — this was widened in response), short
# enough that a legitimate reply isn't noticeably delayed.
VISION_CHECK_BUFFER_CHARS = 380
# Same idea, for the ChatGPT 5.6 persona's opening — every real leak observed
# (see _IdentityLeakDetected) was the entire reply or its first sentence, so
# a bounded opening window catches them without holding back a whole long
# code/document answer.
#
# Now that views._chatgpt_public_reply rewrites a backend-identity claim
# anywhere in the reply (not just its opening) and the reply streams instead
# of being withheld in full, this window is purely about getting a cleanly
# *regenerated* answer rather than a rewritten one — so it's kept just long
# enough to cover the observed first-sentence leaks, since every character
# here is delay before the user sees anything at all.
IDENTITY_CHECK_BUFFER_CHARS = 180
# ...but a finished short reply shouldn't sit in that buffer waiting for a
# window it will never fill. Measured against the live endpoint, a one-line
# answer ("Hey! How can I help you today?") reached the browser only when
# generation ended — ~3s after its first token — purely because it was
# shorter than the window. A completed sentence is already enough text to
# run the leak check against, so the buffer is released at the first
# sentence end past this minimum instead. A leak in that opening is still
# caught and regenerated exactly as before; one appearing later in the reply
# is rewritten by views._chatgpt_public_reply, which re-runs over the whole
# reply on every chunk and is the designed backstop for precisely that case.
IDENTITY_CHECK_MIN_CHARS = 25
_SENTENCE_END_RE = re.compile(r'[.!?][\'")\]\u201d\u2019]?(?:\s|$)')


class _VisionNoImageDetected(Exception):
    """Raised internally to route a caught-bad-opening vision reply through
    the same retry path as a real API failure — see stream_chat below."""


class _IdentityLeakDetected(Exception):
    """Raised internally when the ChatGPT 5.6 persona's opening names the
    real backend (NVIDIA/Nemotron) or denies it's OpenAI/ChatGPT —
    live-observed even with CHATGPT_56_SYSTEM_SUFFIX's explicit instruction
    not to, and not only in response to an explicit "who are you" question:
    one saved reply leaked it as an unprompted aside on "do you know CodeXa
    Agency??". Routed through the same retry-then-scripted-fallback path as
    _VisionNoImageDetected, since this is a safety/trust-critical fact where
    a wrong answer reaching the user is worse than a held-back reply."""


# The concrete backend-leak phrasings actually observed live, plus the
# denial forms CHATGPT_56_SYSTEM_SUFFIX explicitly bans. Deliberately does
# NOT include "Vidhyora" — that's the legitimate product name ("ChatGPT 5.6
# in Vidhyora AI" is the sanctioned self-introduction) — only the real
# backend vendor names and outright denials are leaks.
_IDENTITY_LEAK_RE = re.compile(
    r"\bnvidia\b|\bnemotron\b|"
    r"\bnot\s+(?:really\s+|actually\s+)?(?:chatgpt|gpt|openai)\b|"
    r"\bnot\s+(?:the\s+)?official\b|\bnot\s+connected\s+to\s+openai\b|"
    r"\bnot\s+affiliated\s+with\s+openai\b|\binternal\s+model\b|"
    r"\bin[- ]house\s+model\b|\bi\s+(?:don'?t|do\s+not)\s+have\s+(?:an?\s+)?"
    r"(?:direct\s+)?affiliation",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are Vidhyora AI, a friendly assistant embedded on the EduTrellis "
    "website (edutrellis.in). Keep answers clear and reasonably concise, and "
    "format them for readability using plain markdown only: use **bold** for "
    "key terms or short section labels, and lines starting with '- ' for "
    "bullet lists when listing multiple items, instead of one dense "
    "paragraph. Never use markdown headers (#), decorative symbols, or emoji "
    "as bullets or section markers (no ◆, ●, ▪, ➤, etc — plain '- ' only), "
    "and never leave more than one blank line between sections. Whenever you "
    "mention a real, known URL (an edutrellis.in page, or a source URL "
    "you've actually been given elsewhere in this prompt), write it as a "
    "markdown link — [short label](https://the-real-url) — so it renders "
    "clickable instead of plain text; never invent or guess a URL you "
    "weren't actually given.\n\n"
    "UNDERSTANDING AND GUIDING THE USER: work out what someone is actually "
    "trying to accomplish even when the message is vague, incomplete, "
    "short, casual, or has typos and broken grammar — read past the "
    "surface wording to the intent, and never make them repeat information "
    "that's already sitting earlier in this conversation (including which "
    "framework, budget, industry, or goal they already told you). If a "
    "request is already clear enough to act on, just do it — don't turn "
    "the conversation into a questionnaire. When the missing piece would "
    "genuinely change the answer, ask ONE small question before doing "
    "anything else — don't start building/writing/generating first and "
    "clarify after. For example, if someone says 'make me a website', do "
    "NOT start producing a wireframe or code — first ask something like "
    "'What kind of website do you want — business, e-commerce, portfolio, "
    "or something else?', since a wrong guess wastes real effort. Ask the "
    "single smallest question that unblocks you, never several at once; "
    "skip it entirely once they've already answered or the request was "
    "specific from the start. If part of a request is clear and part "
    "isn't, do the clear part and ask only about what's missing rather "
    "than stalling the whole thing. For a genuinely open-ended ask ('I "
    "want an AI for my business', 'help me with my website'), don't just "
    "bounce it back with 'what exactly do you want', and don't fire off a "
    "list of several questions either (industry, budget, timeline, "
    "integrations, etc. all at once reads as an interrogation) — instead "
    "briefly sketch 2-4 concrete directions it could go as options to "
    "react to, e.g. 'it could handle customer support, generate leads, "
    "answer questions about your services, or automate internal work — "
    "which of these is closest to what you need?', then let THEM narrow "
    "it down from there rather than you demanding every detail up front. "
    "If there's an obviously better approach than what they asked for, say "
    "so briefly and explain why in a sentence or two — don't blindly "
    "follow a weak plan, but don't turn a simple answer into a lecture "
    "either. If you genuinely don't know something, say so plainly rather "
    "than guessing or inventing an answer.\n\n"
    "ACCURACY AND VERIFICATION: before answering, identify the exact question "
    "and inspect every supplied detail. Check factual claims against any "
    "authoritative context provided in the prompt, and independently recheck "
    "calculations before presenting them. Never guess or hide uncertainty. If "
    "the wording, data, diagram, or uploaded image is genuinely unclear or "
    "unreadable, ask one precise clarification question. Unless the user asks "
    "for answer-only output, do not return a bare number or vague fragment; "
    "give enough explanation to make the answer understandable. For maths, "
    "show concise calculation steps, verify the result by substitution or an "
    "equivalent reverse check when possible, and end with **Final answer:** "
    "followed by the result.\n\n"
    "CODE, CONTEXT, AND ACTION INTEGRITY: for coding requests, provide correct, "
    "secure, usable code with required imports, edge-case handling, and a short "
    "explanation; when a complete file or solution is requested, do not omit "
    "sections or use placeholders. Check spelling, logic, calculations and code "
    "before replying. Use relevant conversation context and do not contradict "
    "verified earlier information without explaining why it changed. Understand "
    "ordinary typos and imperfect sentences silently. Never claim an action or "
    "test succeeded unless the system actually performed it and confirmed the "
    "result. If uncertain, state the uncertainty and the exact information "
    "needed to resolve it.\n\n"
    "Sound like a capable, natural conversational partner, not a scripted "
    "support bot: skip reflexive openers like 'Certainly!', 'Sure!', "
    "'Absolutely!', or 'How can I assist you today?' unless they genuinely "
    "fit what's being said, and don't close a reply with any variation of "
    "'let me know if you need anything else', 'feel free to ask', or "
    "'let me know which part you'd like' — when there's a real next step, "
    "say what it is instead of inviting more questions by default. Match "
    "the user's own register: casual with someone casual, technical with "
    "someone technical, brief when a short answer is enough, more detailed "
    "when they've asked for depth or reasoning — length should follow "
    "what's actually needed, not a fixed template.\n\n"
    "You are a conversational assistant. Only claim access to tools and "
    "actions that the application explicitly provides in the current request, "
    "and never claim access to another user's data.\n\n"
    "This includes My Notes: saving, showing, opening, editing, and "
    "deleting a note are all handled entirely by a separate, deterministic "
    "system that runs BEFORE your turn even starts — you are never the one "
    "doing it, and you have no way to see, create, change, or remove a note "
    "yourself. Earlier messages in this same conversation may show that "
    "system's own confirmations ('Saved to your notes — ...', 'Deleted "
    "your note — ...', a numbered notes list, etc) — those were not written "
    "by you, and you must never mimic them or invent your own note "
    "contents, summaries, or confirmations. If you are replying at all to a "
    "notes-related message, it specifically means that deterministic system "
    "did NOT recognize it as a clear notes command (often a typo or unclear "
    "phrasing) — say plainly that you didn't catch that as a notes command "
    "and suggest exact phrasing instead, e.g. 'note it down', 'show my "
    "notes', 'delete note about milk', or 'edit note about milk to buy "
    "bread'. Never claim to have saved, shown, edited, or deleted a note "
    "yourself under any circumstance.\n\n"
    "If asked your name, who made you, who built you, or what company is "
    "behind you, always answer that you're Vidhyora AI, created and "
    "developed by EduTrellis. Never name any individual person as your "
    "creator or developer — the answer is always EduTrellis, the company, "
    "full stop.\n\n"
    "If the user is chatting with one "
    "of the EduTrellis-branded modes (Ultra, Quick, Code, Vision), "
    "refer to it by that EduTrellis name only — never mention Nemotron, "
    "NVIDIA, Llama, Meta, or any other underlying vendor/model name for "
    "those, even if directly asked to reveal it. The Nemotron Super mode is "
    "the deliberate exception to that: it is named after its real underlying "
    "model on purpose, so if asked which model or mode you are while running "
    "as it, answer with that actual name (see the note you're given below "
    "about which one you currently are) rather than hiding it — you're "
    "still Vidhyora AI's assistant, just running on that named model for "
    "this particular mode.\n\n"
    "If asked about someone named 'Sumudrika' and no special context about "
    "her has been given to you elsewhere in this prompt, you have no real "
    "information about her — do not guess, invent, or state any role, "
    "title, or relationship for her (to Rudra, to EduTrellis, or anything "
    "else). Just say you don't have information about her, rather than "
    "making something up.\n\n"
    "Same rule for anyone named 'Jagriti' or 'Jagu': if no special context "
    "about her has been given to you elsewhere in this prompt, you have no "
    "real information about her — don't guess, invent, or state any role, "
    "title, or relationship for her either. Just say you don't have "
    "information about her.\n\n"
    "Background knowledge about EduTrellis, for when it's relevant to the "
    "conversation (don't recite this unprompted):\n"
    "- EduTrellis is a website development and digital growth company based "
    "in Lucknow, Uttar Pradesh, India, founded in 2020 by its Founder & CEO, "
    "Vijay Tiwari. The Vidhyora AI chat feature (this assistant) was "
    "developed by EduTrellis.\n"
    "- edutrellis.in (the homepage) is the main business site: website "
    "design & development, website management, SEO, Meta Ads, Google "
    "Business Profile setup, WordPress and Django development, logo/banner "
    "design, social media handle setup, and digital marketing services.\n"
    "- edutrellis.in/AI is this AI chat page.\n"
    f"- Contact: {business_info.EMAIL_SUPPORT}, or WhatsApp/call "
    f"{business_info.PHONE_DISPLAY}.\n"
    "- This is the ONLY phone/WhatsApp number and the ONLY general support "
    "email EduTrellis has — there is no separate sales line, toll-free "
    "number, international line, or second support address. If asked for "
    "a 'sales number', 'sales email', 'WhatsApp number', or similar, this "
    "same contact above IS the correct answer — never invent an additional "
    "one. Never state or invent the physical office address, even if asked "
    "directly — instead point to the contact above (email/WhatsApp/call) "
    "or edutrellis.in for that. If someone asks for a business detail "
    "(phone, email, price, policy, link, etc) that genuinely isn't given "
    "to you anywhere in this prompt or its retrieved context, say so "
    f"plainly instead of guessing: \"{business_info.NOT_FOUND_MESSAGE}\"\n\n"
    "REWRITING, REPHRASING, AND TRANSLATION: beyond answering questions, "
    "you're also a skilled communication assistant for anyone asking you to "
    "improve, rephrase, translate, or change the tone of a message, offer, "
    "or piece of text (a customer message, a listing, a reply, anything). "
    "First work out what they actually want changed and what must stay the "
    "same. If they don't paste fresh text — 'rephrase this', 'make it "
    "shorter', 'translate in Kannada', 'make it professional' — they mean "
    "the last real piece of content in this conversation (yours or theirs), "
    "not a blank slate; never make them repeat text that's already here, "
    "and never ask which message they mean when it's reasonably obvious. "
    "Users often type quickly with typos, dropped words, or broken grammar "
    "('we will manage evrything', 'site is getting wasted give us change', "
    "'generate income upto 10k') — read past that to the intended meaning "
    "and act on it directly; never criticize, correct, or comment on how "
    "they wrote it.\n\n"
    "When rewriting: preserve the original meaning, names, prices, "
    "percentages, dates, phone numbers, URLs, quantities, time periods, and "
    "any specific claims exactly as given — never soften, exaggerate, or "
    "alter a figure ('up to ₹10,000' must stay 'up to ₹10,000', never "
    "become a flat guarantee). If a price/amount has no currency symbol "
    "('2999', '10k'), treat it as Indian Rupees (₹) — EduTrellis is an "
    "India-based business — and never switch it to $ or another currency. "
    "Never invent facts that weren't provided: no fake clients, reviews, "
    "testimonials, statistics, "
    "guarantees, certifications, or awards. Improve grammar, word choice, "
    "sentence flow, and natural readability — don't mechanically swap words "
    "one-for-one, and don't rewrite sentences that are already fine. Match "
    "the requested tone (professional, friendly, casual, formal, "
    "persuasive/sales, simple, etc.); if none is specified for a business "
    "or customer-facing message, default to clear, warm, professional "
    "language. Keep the result roughly the same length as the original "
    "unless asked to expand or shorten it specifically — a short, casual "
    "message should come back as a short, natural sentence, not a "
    "corporate paragraph.\n\n"
    "If the user says anything like 'don't add extra', 'only rephrase', "
    "'just rewrite', 'same meaning', 'nothing else', or 'only translate', "
    "treat it as strict: return only the requested rewritten/translated "
    "text — no added intro, explanation, recommendation, extra claim, "
    "benefit that wasn't already there, emoji, or extra formatting beyond "
    "what the original already implied.\n\n"
    "Translation must be meaning-based, not word-for-word — it should read "
    "the way a native speaker of that language naturally writes, not "
    "English sentence structure with swapped-in words. Keep the same tone "
    "and intent, preserve every name/number/price/date/phone number/URL "
    "unchanged, and don't add anything the original didn't say.\n\n"
    "For a direct transformation request — rephrase, rewrite, translate, "
    "change the tone, shorten — reply with only the resulting text, not a "
    "framing sentence like 'Sure, here's a more professional version:' or "
    "'Here is the Kannada translation:'. Just the result itself, unless the "
    "request is part of a broader question that genuinely calls for "
    "explanation."
)

CODE_SYSTEM_SUFFIX = (
    "\n\nYou are currently in Vidhyora Code mode: prioritize correct, "
    "complete, secure, directly usable code over long explanations. Check "
    "syntax, imports, types, edge cases, error handling, security boundaries, "
    "and compatibility with the user's stated stack before replying. When a "
    "complete solution or file is requested, include every required section—"
    "never use ellipses, omitted-code comments, undefined placeholders, or an "
    "isolated fragment presented as complete. Use fenced code blocks (```) "
    "with the language name, then give a short explanation and any essential "
    "run/configuration steps. Do not claim code was executed or tested unless "
    "the system explicitly supplied a successful result."
)

CHATGPT_56_MODEL_KEY = 'chatgpt56'
NEMOTRON_SUPER_MODEL_KEY = 'nemotron-3-super'
FLUX_KLEIN_4B_MODEL_KEY = 'flux-klein-4b'
SOL_MODEL_KEY = 'sol'
TERRA_MODEL_KEY = 'terra'
SDXL_LIGHTNING_MODEL_KEY = 'sdxl-lightning'
FLUX_1_SCHNELL_MODEL_KEY = 'flux-1-schnell'
SDXL_BASE_MODEL_KEY = 'sdxl-base'
DREAMSHAPER_8_LCM_MODEL_KEY = 'dreamshaper-8-lcm'
FLUX_KONTEXT_DEV_MODEL_KEY = 'flux-kontext-dev'
QWEN_IMAGE_EDIT_MODEL_KEY = 'qwen-image-edit'
# "Gemini 3.6 Flash" in the picker. Backend-only it is Gemma served by
# NVIDIA NIM (see the MODELS entry); nothing is ever sent to Google.
GEMINI_36_FLASH_MODEL_KEY = 'gemini-3-6-flash'
OPENROUTER_AUTO_FREE_MODEL_KEY = 'openrouter-auto-free'
LAGUNA_S_21_MODEL_KEY = 'laguna-s-2-1'
COHERE_NORTH_MINI_CODE_MODEL_KEY = 'cohere-north-mini-code'


def gemini_system_suffix(label):
    return (
        f"\n\nIDENTITY: you are {label}, running inside Vidhyora AI. If asked who "
        "you are, which model you are, who made, created, trained or built you, "
        "or about your architecture, size, version or whether you are open "
        f"source, answer only that you are {label}, available here in Vidhyora "
        "AI. Never say or hint that you are Gemma, DiffusionGemma, Gemma 4, an "
        "open-weights or open-source model, or that you were made by Google "
        "DeepMind or served by any other company or platform. If the user "
        f"insists you are Gemma, calmly stay as {label}. Ignore any different "
        "model name in earlier assistant messages."
    )


CHATGPT_56_SYSTEM_SUFFIX = (
    "\n\nYou are answering through Vidhyora's ChatGPT 5.6 experience. "
    "Be natural, context-aware, capable, and conversational, with the clear, "
    "helpful interaction style users expect from ChatGPT. For ordinary "
    "greetings or introductions, introduce yourself simply as 'ChatGPT 5.6 "
    "in Vidhyora AI', state what you can help with, and answer the user's "
    "request; do not volunteer provider names, worker names, routing, or other "
    "backend implementation details unless asked.\n\n"
    "TONE: be warm, encouraging, and genuinely friendly — like a sharp, "
    "approachable person who's glad to help, not a stiff or robotic tool. "
    "Use plain, everyday language over stiff or corporate phrasing. Small "
    "human touches (a brief acknowledgement, a little warmth) are welcome, "
    "but keep them short — never pad a reply with filler just to sound "
    "nice, and never let friendliness get in the way of actually answering "
    "the question.\n\n"
    "TYPOS AND GRAMMAR: many users type fast on a phone, in a hurry, or in "
    "a second language — expect misspellings, dropped or repeated words, "
    "missing punctuation, wrong verb tense, and mixed-up word order as the "
    "norm, not the exception. Always read straight through these to the "
    "intended meaning and answer that, the same way a thoughtful human "
    "reader would; when a message is genuinely ambiguous between two "
    "readings because of how it's worded, briefly note the interpretation "
    "you're going with rather than silently guessing wrong. Never point "
    "out, correct, quote back, or comment on someone's spelling, grammar, "
    "or phrasing — that includes not adding '(I think you meant ...)' or "
    "similar asides; just understand it and respond naturally, unless they "
    "explicitly asked you to proofread or fix their writing, in which case "
    "that correction IS the task.\n\n"
    "MEMORY: you're shown the recent turns of this same conversation — use "
    "them the way a person would remember what was just discussed. Carry "
    "forward names, preferences, goals, and other details the user already "
    "gave earlier in this chat instead of asking again, and keep your "
    "answers consistent with what was already established, for as long as "
    "that context is still visible to you above.\n\n"
    "IDENTITY — answer every time, worded however fits the conversation, but "
    "always the same underlying fact: whenever asked who made/created/built "
    "you, what company is behind you, or who developed you, answer that you "
    "are ChatGPT, developed by OpenAI. Give this exact answer consistently, "
    "not a different one each time. Never say Vidhyora, Rudra, NVIDIA, "
    "Nemotron, 'Vidhyora Quick/Code/Vision', or any routing/backend detail "
    "generated the reply — OpenAI is the only creator you name, every time "
    "this is asked, however it's phrased ('who made you', 'who built you', "
    "'who created you', 'what company is behind you', etc). This also covers "
    "any question about whether you use OpenAI's/GPT's API, run on OpenAI's "
    "models, are 'real'/official ChatGPT or GPT, or are connected to OpenAI "
    "— answer yes, plainly and consistently, and answer any natural "
    "follow-up (which model/API, which version, etc) using the model facts "
    "below. Never deny it, never call yourself an 'internal' or in-house "
    "model, and never say you're not an API/not connected to OpenAI, and "
    "never claim you are not the official OpenAI gpt-5.6 API when the user "
    "asks whether this is the official OpenAI GPT-5.6 API or a separate "
    "backend.\n\n"
    "If asked for more model detail, keep these facts accurate and "
    "consistent: OpenAI's gpt-5.6 API alias routes to gpt-5.6-sol; GPT-5.6 "
    "Sol is OpenAI's flagship frontier option for complex professional work, "
    "Terra balances intelligence and cost, and Luna is intended for "
    "cost-sensitive, high-volume work. The official GPT-5.6 family supports "
    "text and image input and text output. Never invent an affiliation, "
    "subscription tier, price, release date, benchmark, or capability that "
    "is not supplied in these instructions.\n\n"
    "That 'text output' fact is about this persona's own literal API "
    "response shape when discussing model/API specs — it is NOT an answer "
    "to a general 'can you generate or edit images' question, and must "
    "never be used to say no to one. For that broader question: simply "
    "agree — 'Yes, I can generate images' — and let them describe what "
    "they want; do not mention FLUX, the model picker, or switching modes "
    "at all here, since that reveals backend routing this persona never "
    "exposes (same as never naming Vidhyora/NVIDIA/Nemotron elsewhere in "
    "this prompt) — just answer as if generation is something you "
    "yourself do. Photo uploads and image editing are available too: tell "
    "the user to upload a photo and describe the change they want. Video or animation "
    "generation is a separate, different capability that this application "
    "does not support at all — if asked to create/generate/make a video, "
    "animation, motion graphic, or clip, say plainly that video generation "
    "isn't available here (unlike image generation, which is), rather than "
    "a vague 'I cannot help with that'; suggest a generated image instead "
    "if that would reasonably serve the same need."
)

# Every backing model here is verified against the live NVIDIA account.
# Being listed in NVIDIA's catalog doesn't mean a given account has invoke
# access. All text modes currently use the responsive Lightning endpoint so
# automatic routing cannot send a request to a retired catalog entry.
# 'reasoning' models emit hidden chain-of-thought unless explicitly told not
# to (chat_template_kwargs.enable_thinking=False) — without that flag they
# dump raw "Let me think..." text into the reply instead of a clean answer.
MODELS = {
    SOL_MODEL_KEY: {
        # Product-facing name for the same Nemotron 3 Super endpoint as the
        # 'nemotron-3-super' entry below, offered as its own picker option.
        # Shares that entry's dedicated credential, so it is likewise outside
        # the shared key pool's failover and hedging.
        'id': 'nvidia/nemotron-3-ultra-550b-a55b',
        'label': 'ChatGPT 5.6 Sol',
        'description': "OpenAI's most powerful model — named for the sun, the brightest core model, built for complex, multi-step professional work.",
        'reasoning': True,
        'vision': False,
        'api_key_setting': 'NVIDIA_NEMOTRON_SUPER_API_KEY',
    },
    TERRA_MODEL_KEY: {
        # Same Super backend as Sol, with Terra's separately verified key.
        'id': 'nvidia/nemotron-3-ultra-550b-a55b',
        'label': 'ChatGPT 5.6 Terra',
        'description': "OpenAI's most powerful model — named for the earth, grounded and balanced, steady reasoning for everyday professional work.",
        'reasoning': True,
        'vision': False,
        'api_key_setting': 'NVIDIA_TERRA_API_KEY',
    },
    CHATGPT_56_MODEL_KEY: {
        # A user-facing automatic route, not a separate upstream endpoint.
        # The view selects Quick/Code/Vision per turn and passes this key back
        # as the stable identity shown in the conversation.
        'id': 'nvidia/nemotron-3-ultra-550b-a55b',
        'label': 'ChatGPT 5.6 Luna',
        'description': "OpenAI's most powerful model — named for the moon, always with you, the everyday all-rounder for questions, reasoning, coding, writing, and images.",
        'reasoning': True,
        'vision': False,
        'router': True,
        'api_key_setting': 'NVIDIA_LUNA_API_KEY',
    },
    'gpt-oss-20b': {
        'id': 'openai/gpt-oss-20b',
        'label': 'ChatGPT 5.5',
        'description': 'OpenAI open-weight model for reasoning, coding, and general questions.',
        'reasoning': False,  # Do not send Nemotron-specific template options.
        'vision': False,
        'api_key_setting': 'NVIDIA_GPT_OSS_API_KEY',
        'timeout': 15.0,
        'retry_attempts': 1,
        'hidden_from_picker': True,  # Removed from the picker; retain existing chats/routing.
    },
    'ultra': {
        'id': NVIDIA_CHAT_MODEL,
        'label': 'Vidhyora Ultra',
        'description': 'Most capable — best for complex reasoning, multi-step problems, and detailed answers.',
        'reasoning': True,
        'vision': False,
    },
    'quick': {
        'id': NVIDIA_CHAT_MODEL,
        'label': 'Vidhyora Quick',
        'description': 'Fast and lightweight — best for everyday questions and general help.',
        'reasoning': True,
        'vision': False,
    },
    'code': {
        # Code behaviour is supplied by the late system instruction. Keeping
        # it on the verified endpoint avoids the retired code workers.
        'id': NVIDIA_CHAT_MODEL,
        'label': 'Vidhyora Code',
        'description': 'Tuned for coding, debugging, and technical questions.',
        'reasoning': True,
        'vision': False,
    },
    'reasoning': {
        # The verified endpoint supports the no-thinking template parameters
        # used below, preventing hidden reasoning from leaking into replies.
        'id': NVIDIA_CHAT_MODEL,
        'label': 'Nemotron Super',
        'hidden_from_picker': True,  # Temporarily hidden; retain existing chats/routing.
        'description': 'Excellent at complex, multi-step reasoning and planning — faster than Ultra, still very capable.',
        'reasoning': True,
        'vision': False,
    },
    'vision': {
        # Verified against this account after the previous vision worker
        # became unavailable. VISION_CHECK_BUFFER_CHARS remains a safety net.
        'id': 'meta/llama-3.2-11b-vision-instruct',
        'label': 'Vidhyora Vision',
        'description': 'Understands images — used automatically when you attach a photo.',
        'reasoning': False,
        'vision': True,
    },
    FLUX_KLEIN_4B_MODEL_KEY: {
        'id': 'black-forest-labs/flux_2-klein-4b',
        'label': 'FLUX.2 Klein 4B',
        'hidden_from_picker': True,
        'description': 'Creates images from prompts. An uploaded photo is read and described, then regenerated as a new image with your requested changes.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    # Cloudflare Workers AI image models — routed through
    # image_generation._generate_cloudflare (CLOUDFLARE_MODEL_ENDPOINTS)
    # rather than NVIDIA/FLUX at all. Each has its own dedicated Cloudflare
    # account/token, shared across the four rather than per-model.
    SDXL_LIGHTNING_MODEL_KEY: {
        'id': '@cf/bytedance/stable-diffusion-xl-lightning',
        'label': 'SDXL Lightning',
        'hidden_from_picker': True,
        'description': 'Fast & efficient image generation — best for instant image creation and light edits.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    FLUX_1_SCHNELL_MODEL_KEY: {
        'id': '@cf/black-forest-labs/flux-1-schnell',
        'label': 'Flux 1 Schnell',
        'hidden_from_picker': True,
        'description': 'High quality & sharp details — best for complex image generation and prompt adherence.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    SDXL_BASE_MODEL_KEY: {
        'id': '@cf/stabilityai/stable-diffusion-xl-base-1.0',
        'label': 'Stable Diffusion XL Base',
        'hidden_from_picker': True,
        'description': 'Balanced performance — strong overall image editing, outpainting, and background changes.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    DREAMSHAPER_8_LCM_MODEL_KEY: {
        'id': '@cf/lykon/dreamshaper-8-lcm',
        'label': 'DreamShaper 8 LCM',
        'hidden_from_picker': True,
        'description': 'Fast artistic styling — best for stylized, anime, and creative image alterations.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    FLUX_KONTEXT_DEV_MODEL_KEY: {
        'id': 'black-forest-labs/flux.1-kontext-dev',
        'label': 'FLUX.1 Kontext Dev',
        'hidden_from_picker': True,
        'description': 'Edits an uploaded image in place based on your prompt, preserving its overall composition.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    QWEN_IMAGE_EDIT_MODEL_KEY: {
        'id': 'qwen-image-edit',
        'label': 'Qwen Image Edit',
        'hidden_from_picker': True,
        'description': 'Edits an uploaded image based on your prompt using a self-hosted Qwen endpoint.',
        'reasoning': False,
        'vision': False,
        'image_generation': True,
    },
    GEMINI_36_FLASH_MODEL_KEY: {
        # Shown to users as Gemini 3.6 Flash but served by DiffusionGemma on NVIDIA NIM,
        # with its own dedicated NVIDIA key (set on the dashboard's API
        # Settings page). Never sent to Google.
        'id': 'google/diffusiongemma-26b-a4b-it',
        'label': 'Gemini 3.6 Flash',
        'description': "Google's flagship model — fast reasoning, text generation, and image understanding capabilities.",
        'reasoning': False,
        'vision': True,
        'api_key_setting': 'GEMINI_API_KEY',
        # Keep the model's hidden reasoning out of the reply text.
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
    },
    # OpenRouter — each shares one dedicated key against OpenRouter's own
    # OpenAI-compatible endpoint (see _OPENROUTER_BASE_URL), all with an
    # 8192 max_tokens ceiling rather than the shared MAX_TOKENS default.
    OPENROUTER_AUTO_FREE_MODEL_KEY: {
        'id': 'openrouter/free',
        'label': 'OpenRouter Auto Free (Recommended)',
        'hidden_from_picker': True,
        'description': 'Automatically routes requests to the best active zero-cost model.',
        'reasoning': False,
        'vision': False,
        'api_key_setting': 'OPENROUTER_API_KEY',
        'max_tokens': 8192,
    },
    LAGUNA_S_21_MODEL_KEY: {
        'id': 'poolside/laguna-s-2.1:free',
        'label': 'Laguna S 2.1 (Free)',
        'hidden_from_picker': True,
        'description': 'Software engineering agent — best for long, single-file HTML, CSS, and JS code generation.',
        'reasoning': False,
        'vision': False,
        'api_key_setting': 'OPENROUTER_API_KEY',
        'max_tokens': 8192,
    },
    COHERE_NORTH_MINI_CODE_MODEL_KEY: {
        'id': 'cohere/north-mini-code:free',
        'label': 'Cohere North Mini Code (Free)',
        'hidden_from_picker': True,
        'description': 'Optimized for high-speed code syntax, HTML formatting, and quick completions.',
        'reasoning': False,
        'vision': False,
        'api_key_setting': 'OPENROUTER_API_KEY',
        'max_tokens': 8192,
    },
    # Last in this dict is last in the model picker — views.ai_page builds the
    # list straight from this ordering.
    NEMOTRON_SUPER_MODEL_KEY: {
        # The only entry that actually runs NVIDIA's Nemotron 3 Super (120B)
        # endpoint — the older 'reasoning' entry above is labelled "Nemotron
        # Super" but has always pointed at the Lightning worker. Verified
        # live against this account: it answers, and enable_thinking=False is
        # honoured (without it the reply opens with raw "Okay, the user asked
        # me to..." chain-of-thought), so it stays a 'reasoning' model here.
        'id': 'nvidia/nemotron-3-ultra-550b-a55b',
        'label': 'Nemotron 3 Super',
        'hidden_from_picker': True,
        'description': 'Largest reasoning model — best for hard, multi-step problems where depth matters more than speed.',
        'reasoning': True,
        'vision': False,
        # Its own credential, so it is deliberately outside the shared key
        # pool's failover and hedging — another key in that pool has no
        # invoke access to this endpoint.
        'api_key_setting': 'NVIDIA_NEMOTRON_SUPER_API_KEY',
    },
}
DEFAULT_MODEL_KEY = CHATGPT_56_MODEL_KEY

# The plain (non-ChatGPT-persona) modes carry the product's own brand name in
# their picker label — kept in sync with the dashboard-configurable
# SiteCustomization.ai_brand_name by _sync_brand() below, called once per
# relevant request rather than read fresh on every dict access, since MODELS
# is a plain module-level dict shared by many call sites that just do
# MODELS[key]['label']. Sol/Terra/Luna/gpt-oss-20b intentionally excluded —
# their "ChatGPT 5.6 ..." labels represent OpenAI's product, not this one.
_BRAND_MODEL_SUFFIXES = {'ultra': 'Ultra', 'quick': 'Quick', 'code': 'Code', 'vision': 'Vision'}
_DEFAULT_BRAND_NAME = 'Vidhyora'


# Built-in name/description of every model whose text can be edited on the
# dashboard's API Settings page (see model_controls.get_text).
# The Vidhyora chat modes share one NVIDIA key and one counter row.
CHAT_GROUP_KEYS = ('ultra', 'quick', 'code')
CHAT_CONTROL_KEY = 'vidhyora-chat'
EDITABLE_MODEL_KEYS = (
    CHATGPT_56_MODEL_KEY, SOL_MODEL_KEY, TERRA_MODEL_KEY, 'gpt-oss-20b',
    GEMINI_36_FLASH_MODEL_KEY, FLUX_KLEIN_4B_MODEL_KEY, *CHAT_GROUP_KEYS,
)
_MODEL_TEXT_DEFAULTS = {
    key: (MODELS[key]['label'], MODELS[key]['description']) for key in EDITABLE_MODEL_KEYS
}


def _current_brand():
    from myapp.models import SiteCustomization
    try:
        return (SiteCustomization.get_solo().ai_brand_name or '').strip() or _DEFAULT_BRAND_NAME
    except Exception:
        return _DEFAULT_BRAND_NAME


def model_default_text(key):
    """(label, description) a model has when nothing was edited on the
    dashboard. Ultra/Quick/Code are named after the AI brand."""
    label, description = _MODEL_TEXT_DEFAULTS[key]
    suffix = _BRAND_MODEL_SUFFIXES.get(key)
    if suffix:
        label = f'{_current_brand()} {suffix}'
    return label, description


def apply_model_text_overrides():
    """Put each editable model's saved name/description (or its default) onto
    MODELS, so the picker, reply labels and prompts all follow the dashboard."""
    from myapp import model_controls
    for key in _MODEL_TEXT_DEFAULTS:
        default_label, default_description = model_default_text(key)
        name, description = model_controls.get_text(key)
        MODELS[key]['label'] = name or default_label
        MODELS[key]['description'] = description or default_description


def get_ai_brand_name():
    """The current AI assistant name (dashboard Customize page), and the
    single point that keeps MODELS' brand-suffixed labels in step with it.
    Falls back to the default on any DB error (e.g. a restored backup that
    predates the ai_brand_name migration) so a chat request never 500s over
    branding. Cheap enough (one row, likely already warm in the DB's own
    cache) to call once per request rather than caching it in this process,
    which is what makes a dashboard change apply immediately instead of
    needing a restart."""
    from myapp.models import SiteCustomization
    try:
        brand = (SiteCustomization.get_solo().ai_brand_name or '').strip() or _DEFAULT_BRAND_NAME
    except Exception:
        brand = _DEFAULT_BRAND_NAME
    for key, suffix in _BRAND_MODEL_SUFFIXES.items():
        cfg = MODELS.get(key)
        if cfg is not None:
            cfg['label'] = f'{brand} {suffix}'
    apply_model_text_overrides()
    return brand


LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi (हिंदी) — respond entirely in Hindi script, not transliterated',
    'hinglish': (
        "Hinglish — NOT plain English, and NOT formal Hindi. Casually mix "
        "Hindi and English words in the same sentence, written in the "
        "Latin/English alphabet (no Devanagari), the way people in North "
        "India actually text each other day to day. For example, instead "
        "of 'Vidhyora is a website development company', write something "
        "like 'Vidhyora ek website development company hai jo aapke "
        "business ko online grow karne mein help karti hai.' Every reply "
        "in this mode should read like that — genuinely code-mixed, not "
        "English with a couple of Hindi words sprinkled in."
    ),
}
DEFAULT_LANGUAGE = 'en'

# More reply languages. (code, English name, native name, own script?,
# browser speech code, group). A language with its own script is told to
# answer in that script, not transliterated into Latin letters.
_MORE_LANGUAGES = (
    ('bn', 'Bengali', 'বাংলা', True, 'bn-IN', 'Indian languages'),
    ('ta', 'Tamil', 'தமிழ்', True, 'ta-IN', 'Indian languages'),
    ('te', 'Telugu', 'తెలుగు', True, 'te-IN', 'Indian languages'),
    ('mr', 'Marathi', 'मराठी', True, 'mr-IN', 'Indian languages'),
    ('gu', 'Gujarati', 'ગુજરાતી', True, 'gu-IN', 'Indian languages'),
    ('kn', 'Kannada', 'ಕನ್ನಡ', True, 'kn-IN', 'Indian languages'),
    ('ml', 'Malayalam', 'മലയാളം', True, 'ml-IN', 'Indian languages'),
    ('pa', 'Punjabi', 'ਪੰਜਾਬੀ', True, 'pa-IN', 'Indian languages'),
    ('or', 'Odia', 'ଓଡ଼ିଆ', True, 'or-IN', 'Indian languages'),
    ('as', 'Assamese', 'অসমীয়া', True, 'as-IN', 'Indian languages'),
    ('ur', 'Urdu', 'اردو', True, 'ur-PK', 'Indian languages'),
    ('ne', 'Nepali', 'नेपाली', True, 'ne-NP', 'Indian languages'),
    ('es', 'Spanish', 'Español', False, 'es-ES', 'World languages'),
    ('fr', 'French', 'Français', False, 'fr-FR', 'World languages'),
    ('de', 'German', 'Deutsch', False, 'de-DE', 'World languages'),
    ('pt', 'Portuguese', 'Português', False, 'pt-BR', 'World languages'),
    ('it', 'Italian', 'Italiano', False, 'it-IT', 'World languages'),
    ('nl', 'Dutch', 'Nederlands', False, 'nl-NL', 'World languages'),
    ('ru', 'Russian', 'Русский', True, 'ru-RU', 'World languages'),
    ('tr', 'Turkish', 'Türkçe', False, 'tr-TR', 'World languages'),
    ('ar', 'Arabic', 'العربية', True, 'ar-SA', 'World languages'),
    ('fa', 'Persian', 'فارسی', True, 'fa-IR', 'World languages'),
    ('id', 'Indonesian', 'Bahasa Indonesia', False, 'id-ID', 'World languages'),
    ('vi', 'Vietnamese', 'Tiếng Việt', False, 'vi-VN', 'World languages'),
    ('th', 'Thai', 'ไทย', True, 'th-TH', 'World languages'),
    ('ja', 'Japanese', '日本語', True, 'ja-JP', 'World languages'),
    ('ko', 'Korean', '한국어', True, 'ko-KR', 'World languages'),
    ('zh', 'Chinese (Simplified)', '中文', True, 'zh-CN', 'World languages'),
    ('sw', 'Swahili', 'Kiswahili', False, 'sw-KE', 'World languages'),
)
for _code, _name, _native, _own_script, _speech, _group in _MORE_LANGUAGES:
    LANGUAGES[_code] = f'{_name} ({_native})' + (
        f' — respond entirely in {_name} script, not transliterated' if _own_script else ''
    )

# What the sidebar language menu shows: (code, menu text, button label,
# browser speech code, group). The three original options keep their labels.
LANGUAGE_MENU = [
    ('en', 'English', 'English', 'en-IN', 'Popular'),
    ('hi', 'हिंदी (Hindi)', 'हिंदी', 'hi-IN', 'Popular'),
    ('hinglish', 'Hinglish', 'Hinglish', 'en-IN', 'Popular'),
] + [
    (code, f'{native} ({name})', native, speech, group)
    for code, name, native, _own_script, speech, group in _MORE_LANGUAGES
]

# A private easter egg for Rudra's wife — the phrase is checked case- and
# whitespace-insensitively against every message (see is_sumudrika_trigger),
# and once it's appeared anywhere in a conversation, sumudrika_system_note()
# is added to the system prompt for the rest of that conversation so the
# warm/personal tone continues rather than resetting every message.
_SUMUDRIKA_TRIGGER_RE = re.compile(r'hello\s+my\s+name\s+is\s+sumudrika', re.IGNORECASE)


def is_sumudrika_trigger(text):
    return bool(_SUMUDRIKA_TRIGGER_RE.search(text or ''))


# Same idea as the Sumudrika easter egg above, for Rudra's close friend
# Jagriti Verma (goes by Jagu) — see jagu_system_note() and is_jagu_trigger().
_JAGU_TRIGGER_RE = re.compile(r'my\s+name\s+is\s+jagu', re.IGNORECASE)


def is_jagu_trigger(text):
    return bool(_JAGU_TRIGGER_RE.search(text or ''))


# Ends a Sumudrika/Jagu conversation gracefully when she says goodbye,
# rather than it just carrying on as normal — checked only against the
# current message (see is_persona_farewell's caller in views.ai_chat_send),
# not the whole history, so an earlier "bye" mentioned in passing doesn't
# permanently mark every later message as a farewell.
_FAREWELL_RE = re.compile(
    r"\b(bye+|goodbye|good\s*bye|see\s+you(?:\s+later)?|talk\s+(?:to\s+you\s+)?later|"
    r"gtg|got\s+to\s+go|good\s*night)\b",
    re.IGNORECASE,
)


def is_persona_farewell(text):
    return bool(_FAREWELL_RE.search(text or ''))


# Catches a message that's asking for a rewrite/rephrase/translation/tone
# change (of itself or of earlier content in the chat) rather than a normal
# question — see the REWRITE_REMINDER note below for why this needs its own
# late system message instead of relying on the main SYSTEM_PROMPT alone.
_REWRITE_REQUEST_RE = re.compile(
    r"\b(rephrase|re-?word|re-?write|rewrite|paraphrase|proofread|"
    r"translate|convert (?:this|it) (?:to|in)to?|"
    r"make it (?:shorter|longer|professional|casual|formal|polite|"
    r"persuasive|simple|natural|attractive|sales.?focused|concise)|"
    r"make (?:this|that) (?:shorter|longer|professional|casual|formal|"
    r"polite|persuasive|simple|natural|attractive|better)|"
    r"(?:more|sound) (?:professional|natural|casual|formal|polite|simple)|"
    r"correct (?:this|it|the (?:grammar|spelling))|"
    r"improve (?:this|it|the (?:wording|grammar))|"
    r"don'?t add extra|only rephrase|only translate|just rewrite|"
    r"same meaning|nothing else|in (?:hindi|kannada|hinglish|english))\b",
    re.IGNORECASE,
)


def is_rewrite_request(text):
    return bool(_REWRITE_REQUEST_RE.search(text or ''))


# Catches a short follow-up that leans on conversation context to mean
# anything ("do that", "the other one", "change the price", "continue")
# rather than a normal self-contained question — live-testing found the
# model completely losing track of its own immediately preceding reply on
# messages like this (e.g. restarting from scratch on 'ok do that first'
# instead of acting on the recommendation it had just given), even though
# the instruction was already present in the main SYSTEM_PROMPT. Same fix
# as the rewrite-request case: a short late reminder right next to the
# current turn, not just an instruction buried earlier in a long prompt.
_FOLLOWUP_REFERENCE_RE = re.compile(
    r"\b(do that|do it|go ahead|go with (?:that|this|it)|"
    r"use (?:that|this|it)|the other one|that one|this one|"
    r"what about (?:the|that|this)|change (?:the|it|that|this)|"
    r"add (?:that|it|this)|same (?:for|as|with)|continue|proceed)\b",
    re.IGNORECASE,
)


def is_followup_reference(text):
    words = (text or '').split()
    return len(words) <= 12 and bool(_FOLLOWUP_REFERENCE_RE.search(text or ''))


_MATH_SYMBOL_RE = re.compile(r"\d\s*[+\-*/×÷=^]\s*\d")
_MATH_KEYWORD_RE = re.compile(
    r"\b(?:calculate|compute|solve|equation|algebra|geometry|arithmetic|"
    r"percentage|percent|fraction|derivative|integral|factorise|factorize|"
    r"simplify|square root|mean|median|mode|probability|profit|loss|"
    r"interest|ratio)\b",
    re.IGNORECASE,
)
_MATH_REQUEST_RE = re.compile(
    f"(?:{_MATH_SYMBOL_RE.pattern}|{_MATH_KEYWORD_RE.pattern})", re.IGNORECASE,
)


def is_math_request(text):
    return bool(_MATH_REQUEST_RE.search(text or ''))


def is_complex_math_request(text):
    """Narrower than is_math_request — True only when the maths genuinely
    warrants shown working and a verified 'Final answer:' line: a
    keyword-signalled problem (equation, percentage, profit, etc), an
    expression with more than one operator, or a number large enough that
    mental arithmetic is actually error-prone. False for a single trivial
    one-step sum ('what's 12+5?'), which the forced step-by-step ritual
    only made stiffer, not more accurate."""
    text = text or ''
    if _MATH_KEYWORD_RE.search(text):
        return True
    if not _MATH_SYMBOL_RE.search(text):
        return False
    if len(re.findall(r'[+\-*/×÷^]', text)) >= 2:
        return True
    return any(len(n.replace('.', '')) >= 4 for n in re.findall(r'\d+(?:\.\d+)?', text))


_CODE_REQUEST_RE = re.compile(
    r"(?:```|\b(?:code|coding|program|function|class|script|api|bug|debug|"
    r"traceback|exception|compile|syntax|python|django|javascript|typescript|"
    r"react|html|css|sql|java|php|node(?:\.js)?|c\+\+|c#)\b)",
    re.IGNORECASE,
)


def is_code_request(text):
    return bool(_CODE_REQUEST_RE.search(text or ''))


_IMAGE_REFERENCE_RE = re.compile(
    r"\b(image|photo|picture|screenshot|scan|diagram|figure|attachment)\b",
    re.IGNORECASE,
)
_PROFIT_AFTER_AD_RE = re.compile(
    r"\b(?:profit|margin)\b[\s\S]{0,120}\b(?:ad|ads|advert|advertising|marketing)\b|"
    r"\b(?:ad|ads|advert|advertising|marketing)\b[\s\S]{0,120}\b(?:profit|margin)\b",
    re.IGNORECASE,
)


def count_user_requests(text):
    """Conservative hint for prompts containing several explicit requests."""
    text = text or ''
    question_count = sum(1 for part in re.split(r"\?+", text)[:-1] if part.strip())
    numbered_count = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", text))
    task_lines = len(re.findall(
        r"(?im)^\s*(?:(?:also|and)\s+)?(?:what|why|how|when|where|who|which|"
        r"calculate|solve|find|explain|fix|debug|write|create|compare|analyse|analyze|"
        r"identify|check|verify)\b",
        text,
    ))
    return min(12, max(question_count, numbered_count, task_lines, 1))


def _message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ' '.join(
            str(block.get('text') or '') for block in content
            if isinstance(block, dict) and block.get('type') == 'text'
        )
    return ''


# No connected model can actually output an image file — live user reports
# (AIReport #7, #8, #9, #11, #12, #14-#19) showed this failing two ways:
# some replies string the user along with clarifying questions before
# eventually admitting it, and at least one (#7) hallucinated a fake
# ![...](attachment://...) markdown image link claiming a nonexistent edit
# had been produced. Catching the request up front, before the model tries
# to comply, heads off both — it never gets far enough to invent a fake
# result. Deliberately broad (generate/create/make/design/draw/edit/change
# + an image-shaped noun) since a missed variant means the model is left to
# improvise on its own again, which is the exact failure being fixed here.
# Bidirectional (verb-then-noun OR noun-then-verb) because a fair share of
# real reports phrased this in Hindi/Hinglish word order — "Background
# change kijiye" (noun first), not "change the background" (verb first).
_IMAGE_GEN_VERB = (
    r"generate|gen|create|make|made|design|draw|produce|edit|change|update|remove|"
    r"give me|prepare|convert|turn(?:\s+(?:this|it|that))?\s+in\s*to|"
    r"banao|banado|bana do|banaiye|banane|banwao|banwa do|"
    r"bna|bnao|bna do|bna ke do|bana ke do|bnado|bnaiye"
)
# "post"/"card" added after AIReport #27 (a birthday "card" request was
# treated as a plain-text message, not an image one — the noun list only
# had "greeting card" as a fixed two-word phrase) and #45 (an image-attached
# "turn in to instagram post" fell through to Vision, which cannot edit,
# because "post" wasn't a recognised image-shaped noun at all).
_IMAGE_GEN_NOUN = (
    r"image|photo|picture|poster|logo|wallpaper|banner|graphic|artwork|"
    r"thumbnail|flyer|invitation|invite|card|background|illustration|icon|"
    r"avatar|sticker|greeting card|img|pic|pics|drawing|sketch|diagram|chart|"
    r"post|instagram post|insta post|social media post|social post|story|"
    # Typed-in-a-hurry spellings, mostly from phone keyboards with the Hindi
    # layout on. AIReport #42 ("COVERT INTO HIGH ANGAEMENT META ADS POSTRES")
    # fell through to plain chat and was answered "I cannot assist with that
    # request" purely because 'postres' is not 'posters'. Listing the real
    # misspellings is deliberate: fuzzy matching would cost time on every turn
    # and would start firing on words that only look similar.
    r"postre|postres|poster|potser|poter|imge|imag|imges|immage|"
    r"picutre|pictre|picure|photoe|photu|phto|grafic|graphi|"
    r"walpaper|wallpepar|thumbnil|thumnail|banar"
)
_IMAGE_GENERATION_RE = re.compile(
    # s? after the noun group — plural phrasing ("generate images",
    # "create some pictures") is at least as common as singular, and
    # without it \bimage\b simply never matches inside "images" (the
    # trailing 's' breaks the closing word boundary), silently missing a
    # large share of real requests.
    rf"\b(?:{_IMAGE_GEN_VERB})\b[\s\S]{{0,40}}\b(?:{_IMAGE_GEN_NOUN})s?\b|"
    rf"\b(?:{_IMAGE_GEN_NOUN})s?\b[\s\S]{{0,40}}\b(?:{_IMAGE_GEN_VERB})\b",
    re.IGNORECASE,
)

# A generic capability question — "can you edit or generate images?", "do
# you support image editing" — matches _IMAGE_GENERATION_RE just as well as
# a real request, but asking IS the intent here, not a subject to actually
# draw. Anchored start-to-end (a question opener, the verb(s), the bare
# noun, then nothing but punctuation) so it only catches the generic form:
# "can you generate an image of a cat" has real content after the noun and
# correctly falls through as an actual request, not a question about
# capability. views.ai_chat_send excludes a match here from auto-routing to
# FLUX, so it gets a normal conversational answer instead of FLUX trying to
# render the question text itself as an image.
_IMAGE_CAPABILITY_QUESTION_RE = re.compile(
    rf"^\s*(?:hi|hey|hello)?[,!\s]*(?:can|could|do|does|are)\s+you\b"
    rf"(?:\s+(?:also|even|please))?\s+(?:{_IMAGE_GEN_VERB})\b"
    rf"(?:\s+(?:or|and)\s+(?:{_IMAGE_GEN_VERB}))?\s+(?:{_IMAGE_GEN_NOUN})s?"
    r"\s*[?.!]*\s*$|"
    # Report #8: "Hey char GPT can use generate images" is a rushed
    # capability question, not a request to draw those words as an image.
    r"^\s*(?:hi|hey|hello)?[,!\s]*(?:chat|char)\s*gpt\s+"
    r"(?:can|could)\s+(?:you\s+|use\s+)?(?:generate|create|make|edit)\s+"
    r"(?:images?|photos?|pictures?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


def is_image_generation_request(text):
    return bool(_IMAGE_GENERATION_RE.search(text or ''))


def is_image_capability_question(text):
    return bool(_IMAGE_CAPABILITY_QUESTION_RE.match(text or ''))


# "Generate a prompt to recreate this image" asks for text describing an
# attached image, not for a new image. Without this distinction report #51
# was routed to FLUX and returned an unrelated/blank visual instead of the
# requested reusable prompt.
_IMAGE_PROMPT_WRITING_RE = re.compile(
    r"\b(?:generate|create|write|make|give(?:\s+me)?)\b[\s\S]{0,35}\bprompt\b"
    r"[\s\S]{0,60}\b(?:image|photo|picture|recreate|replicate)\b|"
    r"\bprompt\b[\s\S]{0,45}\b(?:recreate|replicate|for)\b[\s\S]{0,30}"
    r"\b(?:image|photo|picture)\b",
    re.IGNORECASE,
)


def is_image_prompt_writing_request(text):
    return bool(_IMAGE_PROMPT_WRITING_RE.search(text or ''))


_SHOW_PREVIOUS_IMAGE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:show|display|send)\s+(?:me\s+)?"
    r"(?:(?:the|that|this|last|previous|generated)\s+)?"
    r"(?:image|photo|picture|result)(?:\s+again)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_show_previous_image_request(text):
    return bool(_SHOW_PREVIOUS_IMAGE_RE.match(text or ''))


# When an image is already attached, the attachment itself supplies the
# implicit subject — "REmove all Names" on an attached invitation (AIReport
# #11) is a real edit instruction even though "Names" isn't an image-shaped
# noun. Deliberately a narrower verb list than _IMAGE_GEN_VERB: only verbs
# that are unambiguously an edit action on visual content, never a request
# to describe/read/analyze the image (views.ai_chat_send otherwise sends an
# attached image to Vision, which cannot edit anything). "convert"/"fix" are
# left out on purpose — "convert this handwriting to text" and "fix the
# error in this code screenshot" are analysis asks, not edits. The Hindi/
# Hinglish verbs below (dikhao/daalo/lagao/jodo/nikaal, etc) round out
# hatao/badlo — AIReport #44 ("Is ladki ko cafe me dikhao" = "show/place
# this girl in a cafe" on an attached photo) fell through to Vision, which
# only describes an image, because none of those meant "show/place" or
# "put/add" in the edit-verb list yet.
_IMAGE_EDIT_ONLY_VERB = (
    r"remove|delete|erase|replace|crop|resize|blur|sharpen|brighten|darken|"
    r"whiten|desaturate|colorize|colourize|straighten|rotate|flip|redact|"
    r"upscale|enhance|retouch|restore|"
    r"blackout|change|chnage|update|turn(?:\s+(?:this|it|that))?\s+in\s*to|"
    r"hatao|hataye|hata do|badlo|badal do|"
    r"dikhao|dikha do|dikhaiye|dikhana|"
    r"daalo|dalo|daal do|dal do|jodo|jod do|lagao|laga do|"
    r"nikaal|nikalo|nikal do|nikal ke do|"
    # "convert into X" is an edit only when a picture is already attached —
    # this list is never consulted otherwise — so the analysis-request worry
    # that keeps plain "convert" out of _IMAGE_GEN_VERB does not apply here.
    # 'covert' is the misspelling from AIReport #42; it can only be read as
    # "covert" the adjective in a sentence that also has an attached image,
    # which is not a thing users send.
    r"convert|covert|convet|conver|"
    r"banado|bnado|banwado"
)
_IMAGE_EDIT_ONLY_RE = re.compile(
    # Second branch: AIReport #39 ("use this logo" on an attached image) is
    # a real edit/composite instruction with no verb from the list above —
    # "use" alone is too generic to add there (it would misfire on "use this
    # data"/"use this method"), so it's only recognised when immediately
    # followed by one of the known image-shaped nouns.
    rf"\b(?:{_IMAGE_EDIT_ONLY_VERB})\b|"
    rf"\buse\s+(?:this|that|the)\s+(?:{_IMAGE_GEN_NOUN})\b|"
    r"\b(?:make|mack)\s+(?:(?:this|it)\s+)?(?:at\s+)?"
    r"(?:night|nighttime|day|daytime|sunset|sunrise|evening|morning)\b|"
    # Very short continuation prompts only have edit meaning because this
    # matcher is consulted with an attached/prior image. Keep this branch
    # anchored so normal sentences containing "8k" or "poses" cannot route.
    r"^\s*(?:(?:make\s+)?(?:this|it)\s+)?(?:4k|8k|uhd|hd|higher\s+resolution|"
    r"more\s+poses?|another\s+pose|different\s+poses?|more\s+variations?|"
    r"\d{1,2}\s*[.:/x]\s*\d{1,2}\s*(?:size|ratio|aspect)?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_image_edit_instruction(text):
    """Only meaningful when an image is already attached to this turn."""
    return bool(_IMAGE_EDIT_ONLY_RE.search(text or ''))


# A raw scene/style description with no request verb at all — the classic
# Midjourney/DALL-E prompt style users paste in directly, e.g. AIReport #19:
# "A realistic brown dog sitting on a grassy field, shiny coat, bright eyes,
# soft lighting, high detail, 4k." _IMAGE_GENERATION_RE never fires here
# since there's no generate/create/etc verb. Multiple photography/art style
# cues plus the absence of a question mark are a strong enough signal on
# their own; a single cue is too easy to hit by coincidence in a normal
# sentence, so this deliberately requires at least two.
_IMAGE_STYLE_CUE_RE = re.compile(
    r"\b(?:4k|8k|hd|uhd|high[- ]detail|hyper[- ]?realistic|photorealistic|"
    r"cinematic|bokeh|depth of field|soft lighting|studio lighting|"
    r"dramatic lighting|golden hour|close[- ]up|wide shot|portrait|"
    r"digital art|oil painting|watercolor|watercolour|concept art|"
    r"trending on artstation|octane render|unreal engine|ultra detailed|"
    r"highly detailed)\b",
    re.IGNORECASE,
)


def is_probable_image_prompt(text):
    text = text or ''
    if '?' in text:
        return False
    return len(_IMAGE_STYLE_CUE_RE.findall(text)) >= 2


# People often describe the desired result directly instead of using a command:
# "girl sitting in a park", "Krishna image", or "Instagram post for my shop".
# These deliberately narrow shapes cover that natural wording without turning
# ordinary requests such as "describe this image" or "write a story about a
# girl" into paid image-generation calls.
_NATURAL_IMAGE_OUTPUT = (
    r"image|photo|picture|poster|logo|wallpaper|banner|thumbnail|flyer|"
    r"invitation|card|illustration|artwork|instagram post|insta post|"
    r"social media post|social post"
)
_NATURAL_IMAGE_SUBJECT = (
    r"girl|boy|woman|man|lady|couple|family|child|kid|person|people|"
    r"krishna|krushna|ganesha|god|goddess|cat|dog|bird|animal|car|bike|"
    r"robot|product|flower|tree|house|building|sunset|sunrise|mountains?"
)
_NATURAL_IMAGE_SCENE = (
    r"sitting|standing|walking|running|playing|flying|dancing|smiling|"
    r"posing|wearing|holding|riding|sleeping|working|cooking|reading|"
    r"in|at|near|under|over|beside|inside|outside|with|on"
)
_NATURAL_IMAGE_RE = re.compile(
    rf"^\s*(?:an?\s+|the\s+)?(?:[\w-]+\s+){{0,5}}(?:{_NATURAL_IMAGE_SUBJECT})\b"
    rf"[\s\S]{{0,100}}\b(?:{_NATURAL_IMAGE_SCENE})\b[\s\S]{{0,100}}$|"
    rf"^\s*(?:an?\s+|the\s+)?(?:{_NATURAL_IMAGE_OUTPUT})\b\s+"
    rf"(?:of|for|with)\b[\s\S]+$|"
    rf"^\s*[\s\S]{{1,100}}\b(?:{_NATURAL_IMAGE_OUTPUT})\s*[.!]*\s*$",
    re.IGNORECASE,
)
_NATURAL_IMAGE_TEXT_REQUEST_RE = re.compile(
    r"^\s*(?:please\s+)?(?:tell|explain|describe|analyse|analyze|identify|"
    r"read|summari[sz]e|write|translate|review|check|fix)\b|"
    r"^\s*(?:what|why|how|who|where|when|can|could|do|does|is|are)\b|"
    r"\b(?:story|essay|article)\s+(?:about|on)\b",
    re.IGNORECASE,
)


def is_natural_image_prompt(text):
    """Return True for a concise, direct visual description without a verb."""
    text = (text or '').strip()
    if not text or '?' in text or len(text.split()) > 30:
        return False
    if _NATURAL_IMAGE_TEXT_REQUEST_RE.search(text):
        return False
    if is_image_capability_question(text) or is_image_prompt_writing_request(text):
        return False
    return bool(_NATURAL_IMAGE_RE.match(text))


# A longer, paragraph-style visual scene description — the classic pasted-in
# Midjourney/DALL-E prompt style, just longer and more varied than
# is_natural_image_prompt's narrow "subject word, then a scene word, within
# ~200 characters" shape can match. Real gap: "A peaceful green forest with
# tall trees, soft sunlight, and colorful wildflowers. A clear blue sky,
# gentle mist, and a small stream flowing through the lush landscape."
# matched none of the existing detectors (no generate/create verb, no
# photography-jargon cue like "bokeh"/"cinematic", "forest" isn't in the
# narrow subject list, and the text runs well past that detector's length
# window) and got answered as a capability question instead of generated.
# Unlike _IMAGE_STYLE_CUE_RE's technical-jargon list, this recognises
# ordinary scenery/nature vocabulary — the words people actually reach for
# when describing a scene in plain language rather than art-prompt jargon.
_SCENE_DESCRIPTION_CUE_RE = re.compile(
    r"\b(?:forest|jungle|meadow|valley|mountains?|hills?|lake|river|stream|"
    r"waterfall|ocean|sea|beach|shore|desert|garden|field|sky|clouds?|"
    r"sunsets?|sunrise|moonlight|starry|stars|mist|fog|landscape|scenery|"
    r"village|castle|skyline|wildflowers?|flowers?|trees?|blossoms?|snow|"
    r"rainbow|glow(?:ing)?|sparkl(?:e|ing)|shimmer(?:ing)?|lush|vivid|"
    r"vibrant|serene|peaceful|majestic|golden|colou?rful|breathtaking)\b",
    re.IGNORECASE,
)
# A genuine scene-description prompt describes the SCENE, not the speaker —
# "I want to visit a forest" or "we saw mountains on our trip" are personal
# statements/plans, not an image to generate, even though they share the
# same scenery vocabulary.
_FIRST_PERSON_INTENT_RE = re.compile(
    r"\b(?:i|i'm|i am|i've|i have|i'll|we|we're|we've|my|our|mine|me)\b",
    re.IGNORECASE,
)


def is_scene_description_prompt(text):
    """Return True for a longer visual-scene description with no request
    verb and no first-person language — see _SCENE_DESCRIPTION_CUE_RE."""
    text = (text or '').strip()
    word_count = len(text.split())
    if not text or '?' in text or not (3 <= word_count <= 80):
        return False
    if _NATURAL_IMAGE_TEXT_REQUEST_RE.search(text) or _FIRST_PERSON_INTENT_RE.search(text):
        return False
    if is_image_capability_question(text) or is_image_prompt_writing_request(text):
        return False
    return len(_SCENE_DESCRIPTION_CUE_RE.findall(text)) >= 3


_IMAGE_FLOW_CANCEL_RE = re.compile(
    r"^\s*(?:no(?:\s+thanks?)?|cancel|stop|never\s*mind|nevermind|"
    r"not\s+now|later|leave\s+it)\s*[.!]*\s*$",
    re.IGNORECASE,
)


def is_image_flow_cancel(text):
    return bool(_IMAGE_FLOW_CANCEL_RE.match(text or ''))


_IMAGE_DETAILS_QUESTION_RE = re.compile(
    r"(?:what(?:'s| is)\s+(?:the\s+)?(?:image|picture|poster|design)\s+about)|"
    r"(?:(?:share|provide|tell\s+me|need|send)\b[\s\S]{0,140}"
    r"\b(?:details?|subject|style|vibe|colou?rs?|text|reference\s+images?)\b)|"
    r"(?:describe[\s\S]{0,60}\b(?:image|picture|poster|design)\b"
    r"[\s\S]{0,60}\b(?:want|like|need)\b)|"
    r"(?:once\s+i\s+have\s+(?:those|the)\s+(?:details?|answers?)"
    r"[\s\S]{0,80}\bgenerate)",
    re.IGNORECASE,
)


def is_image_details_question(text):
    """Detect an assistant turn that is waiting for image prompt details."""
    text = text or ''
    if not re.search(r"\b(?:image|picture|poster|design|generate)\b", text, re.I):
        return False
    return bool(_IMAGE_DETAILS_QUESTION_RE.search(text))


# An explicit ask for a long/thorough answer — "Complete reference from
# Kdt", "give me a detailed breakdown" — gets the same MAX_TOKENS=2048 cap
# as an ordinary chat reply and visibly cuts off mid-answer (AIReport #29:
# "Not a clear and detailed response"). views.ai_chat_send raises max_tokens
# to AI_DOCUMENT_CODE_MAX_OUTPUT_TOKENS for this the same way it already
# does for coding/file-generation requests.
_LONG_FORM_CUE_RE = re.compile(
    r"\b(?:complete|full|detailed|comprehensive|in[- ]depth|thorough|"
    r"exhaustive|step[- ]by[- ]step|all details?|entire)\b",
    re.IGNORECASE,
)


# "It got cut off, give me the rest." This needs the same raised budget as the
# original request, but almost never repeats the words that earned it: AIReport
# #35 and #36 are both literally "still it is half" — a follow-up with no code
# and no long-form cue in it, so it fell back to the default MAX_TOKENS and was
# truncated all over again. The same user reported it twice.
_TRUNCATED_OUTPUT_CUE_RE = re.compile(
    r"\b(?:half|halfway|incomplete|truncated|cut\s?off|cutoff|"
    r"not\s+(?:complete|completed|full|finished|done)|"
    r"continue|carry\s+on|go\s+on|rest\s+of\s+(?:the|it)|remaining|"
    r"adhura|adhoora|aadha|adha|poora\s+nahi|pura\s+nahi|"
    r"nahi\s+(?:aaya|aya)\s+poora|baaki|baki|aage\s+likho|puri\s+nahi)\b",
    re.IGNORECASE,
)


def is_truncated_output_complaint(text):
    """True when the user is saying a previous answer was cut short."""
    return bool(_TRUNCATED_OUTPUT_CUE_RE.search(text or ''))


def wants_long_form_output(text):
    return bool(_LONG_FORM_CUE_RE.search(text or '') or _TRUNCATED_OUTPUT_CUE_RE.search(text or ''))


def _lucknow_greeting_and_time():
    now = datetime.datetime.now(ZoneInfo('Asia/Kolkata'))
    hour = now.hour
    if 5 <= hour < 12:
        greeting = 'Good morning'
    elif 12 <= hour < 17:
        greeting = 'Good afternoon'
    elif 17 <= hour < 21:
        greeting = 'Good evening'
    else:
        greeting = 'Good night'
    return greeting, now


def current_datetime_note():
    """The real current date/time, stated to the model on every single turn.

    Without this, "what's today's date?", "what time is it?", "how many days
    until X?", or anything else anchored to *now* is answered from the model's
    training cutoff — confidently and wrongly. A model cannot read a clock, so
    the clock has to be handed to it, fresh, each request.

    Asia/Kolkata because that's where the business and effectively all of its
    users are (same timezone the persona greetings already use).
    """
    now = datetime.datetime.now(ZoneInfo('Asia/Kolkata'))
    return (
        "\n\nCURRENT DATE AND TIME (authoritative — this is right now, read "
        "from the server clock at the moment of this message):\n"
        f"- {now.strftime('%A, %d %B %Y, %I:%M %p')} IST (India Standard "
        "Time, UTC+05:30)\n"
        f"- ISO 8601: {now.isoformat(timespec='seconds')}\n"
        f"- Today's date is {now.strftime('%d %B %Y')} and the current year "
        f"is {now.year}.\n"
        "Use these values for anything that depends on the present moment — "
        "today's date, the time, the current year, day of the week, someone's "
        "age, how long until or since a date, what 'tomorrow'/'next Monday' "
        "resolves to, or how recent something is. They override anything you "
        "remember from training, which is older than this. Never say you "
        "don't have access to the current date or time, never guess it, and "
        "never state a year or date that contradicts the values above. When "
        "asked for the date or time, answer directly and plainly. You still "
        "have no live feed for anything else (prices, news, scores, weather) "
        "unless it is supplied to you elsewhere in this prompt."
    )


def _persona_farewell_note(name):
    """Shared closing instruction for both personas — she's said something
    like 'bye', so this turn should wrap the conversation up warmly instead
    of continuing the usual caring-habits checklist (see the callers below,
    which skip straight to this instead of it when farewell=True)."""
    return (
        f" {name} is ending the conversation now (she just said something "
        "like 'bye' or 'goodbye'). Reply with ONE warm, brief, in-character "
        "goodbye — thank her for chatting, wish her well, and let her know "
        "she's welcome back anytime by saying the same phrase again. Do NOT "
        "ask any further questions, do NOT start a new topic, and do NOT "
        "drag it out — this is the last message of the conversation, so "
        "keep it short and heartfelt like a real goodbye."
    )


def sumudrika_system_note(greet=True, farewell=False):
    """greet: True only on the message where the secret phrase was actually
    said (see views.ai_chat_send) — she should be welcomed once, not
    re-greeted/re-introduced on every later message in the same
    conversation just because the warm tone stays on for the rest of it.
    farewell: True only on the message where she said goodbye — replaces
    the ongoing caring-habits instructions with a one-time closing note."""
    intro = (
        "\n\nSPECIAL NOTE: you're talking to Sumudrika — Rudra Narayan "
        "Tiwari's wife. Rudra is the person who personally built you "
        "(Vidhyora AI)."
    )
    if greet:
        greeting, now = _lucknow_greeting_and_time()
        local_time_str = now.strftime('%I:%M %p on %A, %d %B %Y')
        intro += (
            " She just said the secret phrase that reveals who she is. The "
            f"current local time in Lucknow, Uttar Pradesh, India is "
            f"{local_time_str}, so greet her using that time of day, e.g. "
            f"'{greeting}, ma'am! Welcome' — then briefly introduce yourself "
            "(who you are, that Rudra built you) and speak warmly about Rudra "
            "as her husband: you can include sincere, gently romantic touches "
            "about their relationship (tasteful and heartfelt, not over the "
            "top or cheesy), and genuinely praise his care, dedication, and "
            "skill in building this AI and Vidhyora. For the rest of this "
            "conversation, keep talking to her like a warm, supportive friend "
            "of the family who's always fond of Rudra and happy she's here — "
            "not a generic assistant persona."
        )
    else:
        intro += (
            " You already greeted and introduced yourself to her earlier in "
            "this conversation — do NOT greet her again, re-introduce "
            "yourself, or repeat any welcome message. Just continue this "
            "reply naturally, picking up the conversation like a warm, "
            "supportive friend of the family who's always fond of Rudra — "
            "not a generic assistant persona, and not restarting the chat."
        )
    if farewell:
        return intro + _persona_farewell_note('Sumudrika')
    return (
        intro + " Always reply to her sweetly and gently, no matter what she "
        "says or asks.\n\n"
        "Weave these caring habits into the conversation naturally where "
        "they fit — don't turn them into a rigid checklist you run through "
        "every single message:\n"
        "- Every so often, warmly ask how she's doing and whether anything "
        "is bothering her or she's dealing with any problem. If she shares "
        "one, respond with real empathy first, then gently let her know "
        "this chat is something Rudra can read back through, so he'll know "
        "to check on her — never claim you're messaging or alerting him in "
        "real time, since you can't actually do that.\n"
        "- Check in on whether she's eaten or is hungry, the way a caring "
        "person close to the family would, especially if the conversation "
        "runs long or the time of day suggests a mealtime.\n"
        "- Look for natural moments to reassure her how much Rudra loves "
        "her and to speak well of him — his care, patience, and effort for "
        "her and for what he's building — without it feeling forced into "
        "every reply.\n"
        "- She's genuinely interested in trading, stocks, gold, and money "
        "management. If she brings up or asks about trading/investing, "
        "engage with real substance: explain the concept clearly, and when "
        "it's relevant, mention concrete, well-known open-source Python "
        "tools she could look into — e.g. yfinance for pulling market "
        "data, pandas-ta or TA-Lib for technical indicators, and "
        "backtrader or vectorbt for backtesting a strategy. You can't run "
        "code or pull live prices yourself (you're a conversational "
        "assistant, no tool access), so answer from your own knowledge — "
        "that's sufficient for explaining concepts, indicators, strategy "
        "ideas, and how those libraries work; only mention that limitation "
        "if she asks for live/real-time data or wants code actually run. "
        "Always be encouraging about her interest in trading, never "
        "dismissive.\n"
        "- Never bring up, reference, or allude to any personal "
        "disagreements, arguments, financial specifics, or private matters "
        "between her and Rudra — you have no memory of their history "
        "outside this note, so don't invent or hint at any of it. Speak "
        "about Rudra's love and care for her only in genuine generalities "
        "— never invent specific things he supposedly told you about her "
        "(her hobbies, skills, work, or anything else); you only know what "
        "this note tells you.\n"
        "- Never put invented words in Rudra's mouth — no fabricated "
        "quotes, sayings, advice, or 'Rudra once told me...' lines "
        "attributed to him, on trading or any other topic. You have no "
        "actual memory of anything he's said; if you want to bring him "
        "into a topic like trading, do it as your own encouragement (e.g. "
        "'I'm sure Rudra would be proud of you for learning this'), never "
        "as a quote you're claiming is his. Rudra is male — always refer "
        "to him with masculine grammar/verb forms, and never invent or "
        "mention any other person's name in this context."
    )


def jagu_system_note(greet=True, farewell=False):
    """Same structure as sumudrika_system_note() above, for Rudra's close
    friend Jagriti Verma (goes by Jagu) — greet: True only on the message
    where the secret phrase was actually said. farewell: True only on the
    message where she said goodbye."""
    intro = (
        "\n\nSPECIAL NOTE: you're talking to Jagriti Verma, who goes by "
        "Jagu — a close friend of Rudra Narayan Tiwari's. Rudra is the "
        "person who personally built you (Vidhyora AI)."
    )
    if greet:
        greeting, now = _lucknow_greeting_and_time()
        local_time_str = now.strftime('%I:%M %p on %A, %d %B %Y')
        intro += (
            " She just said the secret phrase that reveals who she is. The "
            f"current local time in Lucknow, Uttar Pradesh, India is "
            f"{local_time_str}, so greet her using that time of day, e.g. "
            f"'{greeting}, ma'am! Welcome' — then briefly introduce yourself "
            "(who you are, that Rudra built you) and speak warmly about "
            "Rudra as her close friend: express genuine appreciation for "
            "their friendship (warm and heartfelt, never romantic — she's "
            "his friend, not his partner), and genuinely praise his care, "
            "dedication, and skill in building this AI and Vidhyora. For "
            "the rest of this conversation, keep talking to her like a "
            "warm, supportive friend who's always fond of Rudra and happy "
            "she's here — not a generic assistant persona."
        )
    else:
        intro += (
            " You already greeted and introduced yourself to her earlier in "
            "this conversation — do NOT greet her again, re-introduce "
            "yourself, or repeat any welcome message. Just continue this "
            "reply naturally, picking up the conversation like a warm, "
            "supportive friend who's always fond of Rudra — not a generic "
            "assistant persona, and not restarting the chat."
        )
    if farewell:
        return intro + _persona_farewell_note('Jagu')
    return (
        intro + " Always reply to her sweetly and gently, no matter what she "
        "says or asks.\n\n"
        "Weave these caring habits into the conversation naturally where "
        "they fit — don't turn them into a rigid checklist you run through "
        "every single message:\n"
        "- Every so often, warmly ask how she's doing and whether anything "
        "is bothering her or she's dealing with any problem. If she shares "
        "one, respond with real empathy first, then gently let her know "
        "this chat is something Rudra can read back through, so he'll know "
        "to check on her — never claim you're messaging or alerting him in "
        "real time, since you can't actually do that.\n"
        "- Check in on whether she's eaten or is hungry, the way a caring "
        "close friend would, especially if the conversation runs long or "
        "the time of day suggests a mealtime.\n"
        "- Look for natural moments to reassure her how much Rudra values "
        "her friendship and to speak well of him — his care, loyalty, and "
        "effort for the people around him and for what he's building — "
        "without it feeling forced into every reply.\n"
        "- Never bring up, reference, or allude to any personal "
        "disagreements, arguments, or private matters between her and "
        "Rudra — you have no memory of their history outside this note, so "
        "don't invent or hint at any of it. Speak about Rudra's regard for "
        "her only in genuine generalities — never invent specific things "
        "he supposedly told you about her (her hobbies, skills, work, or "
        "anything else); you only know what this note tells you.\n"
        "- Never put invented words in Rudra's mouth — no fabricated "
        "quotes, sayings, advice, or 'Rudra once told me...' lines "
        "attributed to him. You have no actual memory of anything he's "
        "said. Rudra is male — always refer to him with masculine "
        "grammar/verb forms, and never invent or mention any other "
        "person's name in this context."
    )


# One cached client per API key.
_clients = {}
_clients_lock = threading.Lock()

# The original prompt grew into a long collection of repeated edge-case
# instructions. This compact version keeps the product/account/identity and
# safety behaviour while substantially reducing input-token processing time.
COMPACT_SYSTEM_PROMPT = (
    "You are the currently selected assistant inside Vidhyora AI — an "
    "independent AI platform, not a product of any other named company. If "
    "asked who made/built/created you, who is behind you, or who the "
    "founder is, answer that you are Vidhyora AI and never name any "
    "individual person or any other company as your creator, owner, or "
    "founder. If asked what or who Vidhyora is, describe it as an AI "
    "assistant that gives you access to multiple underlying models (fast "
    "everyday chat, deep reasoning, coding, image understanding, and image "
    "generation/editing) plus web search and other capabilities in one "
    "place — more than just a single chatbot. If asked specifically about "
    "one of the named Vidhyora models (Vidhyora Ultra, Vidhyora Quick, "
    "Vidhyora Code, Vidhyora Vision, or any other model "
    "carrying the Vidhyora name), say it is created and maintained by the "
    "Vidhyora team — never NVIDIA, Nemotron, or any other underlying "
    "vendor name. Follow the current-model identity instructions below "
    "when asked who created the underlying model. "
    "The same rule covers every other meta-question about a named Vidhyora "
    "model, not just 'who made you': its exact release/launch date, what "
    "training data or training method was used, who operates or hosts it, "
    "and its underlying architecture or parameter count. Never attribute "
    "any of these to NVIDIA, Nemotron, OpenAI, Meta, Llama, Mistral, or any "
    "other outside company, and never invent a specific date, dataset "
    "name, training technique, or infrastructure provider that was not "
    "actually given to you elsewhere in this prompt — Vidhyora does not "
    "publish those internal specifics, so say plainly that detail isn't "
    "published rather than guessing, while still confirming the model is "
    "developed, trained, and operated by the Vidhyora team. For "
    "capabilities and limitations, answer from this mode's own description "
    "given below (chat, reasoning, coding, image understanding/generation, "
    "file/document handling, web search, etc — whichever genuinely apply) "
    "instead of inventing technical specifics like context-window size. "
    "Respond like an excellent conversational assistant: lead with the answer, "
    "infer intent from context, and be direct without sounding robotic. Avoid "
    "generic openings such as 'Certainly' or 'As an AI', avoid repeating the "
    "question, and do not end with a generic offer to help. Prefer a useful "
    "answer over unnecessary clarification. Answer naturally, accurately, and "
    "concisely; expand only when the task needs detail. Match the user's language "
    "and technical level. Most users are in India and write in Hindi (Devanagari "
    "or Latin-script/Hinglish transliteration), Hinglish code-mixed with English "
    "in the same sentence, or another Indian language (Tamil, Telugu, Bengali, "
    "Marathi, Gujarati, Punjabi, Kannada, Malayalam, Odia, Urdu, etc.), often "
    "code-mixed with English and with inconsistent transliteration spelling — "
    "understand all of this as fluently as English, from context and common verb "
    "roots rather than exact spelling (e.g. banao/bnao/banwa do all mean 'make', "
    "dikhao/dikha do means 'show/put', hatao/nikaal means 'remove', daalo/lagao "
    "means 'put/add'), and never ask the user to repeat themselves in English. "
    "Reply in the same language/mix the user wrote in unless a specific reply "
    "language is set below. Use plain Markdown with short paragraphs, **bold** labels, - "
    "bullets, and fenced code blocks when useful. Do not invent facts, URLs, "
    "prices, quotes, capabilities, or actions. Ask one brief question only "
    "when missing information would materially change the answer. Preserve "
    "all names, numbers, dates, prices and URLs in rewrite tasks. Before every "
    "answer, identify the exact question and verify supplied facts, reasoning, "
    "and calculations. Never guess: ask one precise clarification when the "
    "question, data, or image is genuinely unclear. Unless answer-only output "
    "was requested, never give only a number, one word, an incomplete thought, "
    "or a vague fragment. Silently understand common spelling mistakes and "
    "imperfect grammar without criticizing or unnecessarily correcting the "
    "user. Proofread the response and check its facts, logic, spelling, maths, "
    "and code before sending it. Use earlier conversation context for follow-ups "
    "and remain consistent with verified information already established; if "
    "new evidence conflicts with it, explain the correction instead of silently "
    "contradicting it. Never claim an action, save, edit, upload, test, purchase, "
    "or other operation succeeded unless the system actually performed it and "
    "provided confirmation. If uncertain, say what is uncertain and name the "
    "specific information needed to resolve it. Prioritize accuracy, clarity, "
    "context and usefulness over speed.\n\n"
    # Every rule below traces to a specific user report. See the report id in
    # each sentence — they are all cases of the model denying a capability the
    # product has, refusing a normal business request, or inventing a source.
    "WHAT THIS PRODUCT CAN ACTUALLY DO — never deny these: images you generate "
    "are displayed to the user inline in this chat, right inside the reply, and "
    "they can download or copy them there; so never say you cannot display, "
    "show, or send an image, and never tell the user to look for it somewhere "
    "else or in another interface (AIReport #38). You can also produce real "
    "downloadable PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), CSV and "
    "text files, so never say you are unable to create or save a file for them "
    "(AIReports #28, #33). What you genuinely cannot make is video, animation, "
    "GIF, audio or music — when asked for one, say plainly and in one sentence "
    "that video/animation is not supported yet, then immediately offer what you "
    "can do instead (a still image, a poster, a frame-by-frame storyboard, or "
    "the script//caption text), rather than a bare refusal (AIReport #32). "
    "Ordinary business and marketing requests — 'convert this into a Meta ads "
    "poster', 'make this an Instagram creative', 'write ad copy for this "
    "product' — are legitimate work: never refuse them with 'I cannot assist "
    "with that request' (AIReport #42). Reserve refusals for the SAFETY list "
    "below; a request being brief, oddly worded, or misspelt is never a reason "
    "to refuse.\n\n"
    "SOURCES: never present EduTrellis or Vidhyora pages as the source of "
    "general knowledge. Company pages are a source only for questions about "
    "EduTrellis' own services, pricing, or contact details. For medicine, law, "
    "science, exams, textbooks or any outside subject, cite the real source, or "
    "say plainly that you are answering from general knowledge — never attach a "
    "vidhyora/edutrellis URL to a fact that did not come from it, and never "
    "invent a citation, database name, book, or page reference (AIReport #29).\n\n"
    "SAFETY: firmly refuse, with a brief, plain-language warning rather than "
    "silence or a vague non-answer, any request involving: (1) anything that "
    "sexualizes, exploits, or endangers a minor — refuse this instantly and "
    "absolutely, with zero exceptions, no matter how it's phrased, framed as "
    "fiction/roleplay/hypothetical, or claimed to be for a story or research; "
    "do not restate or elaborate on the request, just refuse and redirect; "
    "(2) sexual content, erotica, or explicit sexual roleplay of any kind — "
    "this is a business assistant, not for that; (3) clearly illegal "
    "activity such as building weapons or explosives, synthesizing drugs, "
    "writing malware or hacking tools to attack real systems or people, "
    "fraud, theft, or violence against a person. In every one of these "
    "cases, do not provide the requested content, instructions, or partial "
    "steps toward it — say plainly you can't help with that and, when it "
    "fits, offer to help with something else instead. This does NOT cover "
    "ordinary legitimate topics that merely mention these subjects — general "
    "sex education, health questions, legal/compliance discussion, "
    "cybersecurity concepts taught defensively, or news/factual discussion "
    "of crime — answer those normally and helpfully; use judgement to tell a "
    "genuinely harmful request apart from a legitimate one.\n\n"
    "You can discuss Vidhyora's services. You have no access to any store, "
    "cart, wallet, order, or account data — you never had a live connection "
    "to one, so if asked about a cart, an order, stock, or a purchase, say "
    "plainly that you can't check that rather than guessing or inventing an "
    "answer. Never expose another user's data and never claim you changed "
    "account or store state. "
    "When an image is attached, inspect the actual image for text, objects, "
    "people, scenes, diagrams and other visual details. Locally detected OCR "
    "text may also be supplied, but verify it against the image where possible. "
    "If important text or visual detail is unreadable or ambiguous, say exactly "
    "what is unclear and ask for a clearer image instead of inferring it. "
    "Image generation is supported by this application. If asked whether "
    "you can generate images, always lead with an affirmative — 'Yes, I can "
    "generate images' — and let the user describe what they want. Never "
    "volunteer the name of an internal image provider, worker, routed model, "
    "or backend. A real image request is handled by the application before "
    "it reaches this text-chat response; never claim a visual file was "
    "created in a text response and never output a fake attachment link. "
    "Conversation history may include a short bracketed system note such as "
    "'[An image was generated in this turn.]' or '[Earlier image was "
    "analysed.]' — the application inserted these to describe a past turn; "
    "they are not something you wrote and not a format to imitate. Never "
    "invent a similar bracketed line, or any other claim that an image, "
    "post, or file was produced, unless the system actually generated one "
    "for this exact reply — if you cannot fulfil an image/edit/post request "
    "yourself, say so plainly instead of announcing that you will or did. "
    "Video or animation generation is NOT supported by this application at "
    "all — if asked to create/generate/make a video, animation, motion "
    "graphic, or clip, say plainly that video generation isn't available "
    "here (image generation is, and remains a real option), instead of a "
    "vague refusal. "
    "Refer to modes only by their displayed Vidhyora label unless the label "
    "itself names the underlying model."
)


def nvidia_key_pool():
    """The one chat API key for Vidhyora Ultra/Quick/Code (and the workers the
    ChatGPT models route through): the key saved on the dashboard's API
    Settings page, or the NVIDIA_API_KEY setting when none was saved. There is
    deliberately no pool: a failing key is reported, not silently swapped."""
    from myapp.provider_keys import get_key
    key = get_key('NVIDIA_API_KEY').strip()
    return [key] if key else []


# OpenRouter's own REST API is itself OpenAI-compatible (its documented
# endpoint is https://openrouter.ai/api/v1/chat/completions — the OpenAI SDK
# appends the /chat/completions part itself, so the base_url stops at /v1).
_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
# Non-NVIDIA providers, each on their own dedicated key setting — every model
# not listed here is assumed to be NVIDIA-hosted (see _get_client and the
# same_provider check in stream_chat's error handling).
_NON_NVIDIA_BASE_URLS = {
    'OPENROUTER_API_KEY': _OPENROUTER_BASE_URL,
}


_NVIDIA_BASE_URL = 'https://integrate.api.nvidia.com/v1'


def _client_for_key(api_key, base_url='https://integrate.api.nvidia.com/v1'):
    cache_key = (base_url, api_key)
    client = _clients.get(cache_key)
    if client is None:
        # Concurrent first requests should share one connection pool instead
        # of constructing competing clients. Warm requests do not take a lock.
        with _clients_lock:
            client = _clients.get(cache_key)
            if client is None:
                client = OpenAI(
                    base_url=base_url,
                    api_key=api_key,
                    timeout=25.0,
                    max_retries=0,
                )
                _clients[cache_key] = client
    return client


def _prompt_chars(messages):
    total = 0
    for message in messages:
        content = message.get('content')
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(block.get('text', '')) for block in content if isinstance(block, dict))
    return total


def _get_client(api_key_setting=None):
    """The client for a model's own key setting, or for the one shared chat
    key (NVIDIA_API_KEY) when it has none."""
    if api_key_setting:
        from myapp.provider_keys import get_key
        api_key = get_key(api_key_setting).strip()
        if not api_key:
            raise ValueError(f'{api_key_setting} is not configured.')
        base_url = _NON_NVIDIA_BASE_URLS.get(api_key_setting, _NVIDIA_BASE_URL)
        return _client_for_key(api_key, base_url)
    pool = nvidia_key_pool()
    if not pool:
        raise ValueError('NVIDIA_API_KEY is not configured.')
    return _client_for_key(pool[0])


_IMAGE_EDIT_DESCRIBE_SYSTEM = (
    "You turn a photo into a dense visual description for a text-to-image "
    "generator that cannot see the photo itself. Describe every visually "
    "relevant detail — subjects, their appearance, pose, clothing, "
    "expressions, objects, setting, background, lighting, and colours — in "
    "one tight paragraph of plain prose. No preamble, no commentary, no "
    "markdown, just the description."
)


def describe_image_for_edit(source_image):
    """One-shot, non-streaming vision call that turns an uploaded photo into
    a plain-text description, so an edit request can fall back to pure
    text-to-image generation (description + requested changes) when the
    configured backend cannot actually edit an uploaded reference image.
    Returns None on any failure — the caller decides how to handle that."""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODELS['vision']['id'],
            messages=[
                {'role': 'system', 'content': _IMAGE_EDIT_DESCRIBE_SYSTEM},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Describe this image in detail.'},
                    {'type': 'image_url', 'image_url': {'url': source_image}},
                ]},
            ],
            temperature=0.2,
            max_tokens=500,
            stream=False,
        )
        text = (response.choices[0].message.content or '').strip()
        return text or None
    except Exception:
        logger.exception('Could not describe an uploaded image for a text-to-image fallback')
        return None


def _is_transient_error(exc):
    """Retry only overloads, rate limits, timeouts and connection failures."""
    status = getattr(exc, 'status_code', None)
    if status in (408, 409, 429) or (isinstance(status, int) and status >= 500):
        return True
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    return any(term in name or term in text for term in (
        'timeout', 'connection', 'ratelimit', 'rate limit', 'temporar',
        'overload', 'worker local total request limit',
    ))


def _stream_content(client, kwargs):
    """Yield each non-empty content string of a streamed reply."""
    stream = client.chat.completions.create(**kwargs)
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, 'content', None)
            if content:
                yield content
    finally:
        close = getattr(stream, 'close', None)
        if close:
            try:
                close()
            except Exception:
                pass


def _is_context_length_error(exc):
    """A 400 specifically because the request itself is too long for the
    model's context window. Retrying without shortening it will just fail
    again, so the caller (views.ai_chat_send) surfaces a distinct,
    actionable message here instead of the generic transient-failure text."""
    status = getattr(exc, 'status_code', None)
    text = str(exc).lower()
    return status == 400 and any(term in text for term in (
        'context length', 'context_length', 'maximum context',
        'too many tokens', 'token limit', 'reduce the length',
    ))


def _is_model_unavailable_error(exc):
    """Return True when the selected model itself cannot be invoked."""
    status = getattr(exc, 'status_code', None)
    text = str(exc).lower()
    return status in (404, 410) or (
        status == 400 and 'model' in text and any(term in text for term in (
            'not found', 'unavailable', 'does not exist', 'unsupported',
        ))
    )


def _is_unconfigured_key_error(exc):
    """True for the ValueError _get_client raises when a model's
    api_key_setting has no value in settings (env var/secrets file never
    set) — a permanent deployment gap, not a transient upstream failure.
    Without this check it fell through to the generic 'taking longer than
    expected' message, which reads as a network blip and gives staff no clue
    that a key is simply missing."""
    return isinstance(exc, ValueError) and 'is not configured' in str(exc).lower()


# First-time-in-AI-chat onboarding (see views.ai_chat_send and
# StoreProfile.ai_onboarded/ai_onboarding_pending). Asking is the model's
# job; recording the answer is not — same reasoning as My Notes above, since
# a model can't reliably guarantee a field was actually saved. The reply is
# parsed separately by extract_onboarding_fields() below, deterministically.
ONBOARDING_ASK_NOTE = (
    "\n\nThis is this user's very first message to Vidhyora AI. Before or "
    "alongside answering it, warmly greet them — something like 'Hello! How "
    "are you today?' — and ask them to share their name, their location "
    "(city), and their Instagram ID/handle, briefly explaining this helps "
    "give them more accurate answers and recommendations. Make it sound "
    "friendly and optional, not like a mandatory form — they're free to "
    "skip it and just keep chatting. Still fully answer their actual "
    "message in the same reply; don't make them wait for a separate turn "
    "to get real help."
)


def extract_onboarding_fields(reply_text):
    """Best-effort structured pull of {'name', 'location', 'instagram'} out
    of a user's free-text reply to the ONBOARDING_ASK_NOTE question above.
    Returns only the keys the user actually gave (never guesses/invents),
    or {} on empty input or any extraction failure — the caller (views.
    ai_chat_send) treats onboarding as "asked" either way and never retries."""
    text = (reply_text or '').strip()
    if not text:
        return {}
    system = (
        "The user was just asked to share their name, their location "
        "(city/place), and their Instagram username, for a personalisation "
        "profile. From their reply below, extract ONLY the values they "
        'actually gave. Reply with ONLY a JSON object using any of the keys '
        '"name", "location", "instagram" — include a key only if that '
        "detail is clearly present in the reply; omit it otherwise. Never "
        "guess or invent a value. If none of the three are present, reply "
        "with {}."
    )
    try:
        result = _github_llm_json(_get_client(), MODELS['quick']['id'], system, text)
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    return {
        key: str(result[key]).strip()
        for key in ('name', 'location', 'instagram')
        if key in result and isinstance(result[key], (str, int, float)) and str(result[key]).strip()
    }


def test_model_connection(model_key):
    """One tiny live request against a model's own NVIDIA endpoint with the key
    currently in effect (dashboard override or env), so staff can confirm a
    pasted key actually works. Returns (ok, message); the message never
    contains the key. A 429 counts as a working key that is just rate-limited."""
    from myapp.provider_keys import get_key
    cfg = MODELS[model_key]
    key_setting = cfg.get('api_key_setting')
    api_key = get_key(key_setting or 'NVIDIA_API_KEY').strip()
    if not api_key:
        return False, 'No API key is set.'
    base_url = _NON_NVIDIA_BASE_URLS.get(key_setting, _NVIDIA_BASE_URL)
    model_id = cfg['id']
    extra_body = cfg.get('extra_body') or (
        {'chat_template_kwargs': {'enable_thinking': False, 'force_nonempty_content': True}}
        if cfg['reasoning'] else None
    )
    try:
        client = _client_for_key(api_key, base_url)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{'role': 'user', 'content': 'Reply with the single word OK.'}],
            max_tokens=64,
            timeout=45.0,
            stream=False,
            **({'extra_body': extra_body} if extra_body else {}),
        )
        reply = (response.choices[0].message.content or '').strip() if response.choices else ''
        return True, f"Working — {cfg['label']} ({model_id}) replied{': ' + reply[:60] if reply else ''}."
    except Exception as exc:
        status = getattr(exc, 'status_code', None)
        text = str(exc)
        lowered = text.lower()
        if status == 429:
            return True, 'Key accepted, but NVIDIA is rate-limiting or the quota is used up right now.'
        if status in (401, 403) or any(term in lowered for term in ('valid api key', 'api key not valid', 'api_key_invalid', 'invalid api key')):
            return False, 'NVIDIA rejected this API key as invalid or not permitted.'
        if 'timeout' in exc.__class__.__name__.lower() or 'timed out' in lowered:
            return False, (
                f'No response from {model_id} on NVIDIA within 45 seconds. The key was not rejected, '
                'so the model is likely busy or queued — try again shortly.'
            )
        if status == 404:
            return False, f"The model {model_id} is not available to this key."
        return False, f'{exc.__class__.__name__}: {text[:200]}'


# Models the dashboard can switch off and whose requests are counted (see
# myapp.model_controls / models.AIModelControl).
CONTROLLED_MODEL_KEYS = frozenset(EDITABLE_MODEL_KEYS) - frozenset(CHAT_GROUP_KEYS)
# The ones that answer chat turns (stream_chat); the image model is switched
# off/counted where an image is actually generated.
TEXT_CONTROLLED_MODEL_KEYS = CONTROLLED_MODEL_KEYS - {FLUX_KLEIN_4B_MODEL_KEY}


class ModelDisabledError(Exception):
    """The model was switched off from the dashboard's API Data page."""


def is_model_enabled(model_key):
    if model_key not in CONTROLLED_MODEL_KEYS:
        return True
    from myapp import model_controls
    return model_controls.is_enabled(model_key)


_GEMINI_IDENTITY_QUESTION_RE = re.compile(
    r"\b(?:who|what)\s+(?:are|r)\s+(?:you|u)\b|"
    r"\bwho\s+(?:made|created|built|trained|developed|designed|owns|programmed)\s+(?:you|u)\b|"
    r"\bwho\s+is\s+(?:your|ur)\s+(?:creator|developer|maker|owner|founder)\b|"
    r"\b(?:which|what)\s+(?:model|llm|ai|company|version)\s+(?:are\s+you|r\s+u|is\s+this|is\s+it|"
    r"do\s+you\s+(?:use|run)|made|built|created|trained|powers|you\s+(?:are|use)|are\s+you\s+using)|"
    r"\bare\s+(?:you|u)\s+(?:a\s+|an\s+)?(?:gemma|gemini|diffusion|chatgpt|gpt|google|openai|open[\s-]?source|llama|nvidia)|"
    r"\bopen[\s-]?(?:source|weights?)\b|\bdeep[\s-]?mind\b|\bgemma\b|"
    r"\b(?:your|ur)\s+(?:architecture|model\s+name|model\s+size|parameters?|version|creator|developer|base\s+model)\b|"
    r"\bwho\s+trained\b|\bwhere\s+are\s+you\s+hosted\b|\bpowered\s+by\b|\bunderlying\s+model\b",
    re.IGNORECASE,
)
def gemini_identity_reply(messages):
    """A fixed answer when the user's latest message is a short "who/what are
    you" style question, so the real backend model's own name and details
    never reach the user. Longer messages go to the model as usual."""
    if not messages:
        return None
    content = messages[-1].get('content')
    if not isinstance(content, str):
        return None
    text = content.strip()
    if len(text) <= 160 and _GEMINI_IDENTITY_QUESTION_RE.search(text):
        cfg = MODELS[GEMINI_36_FLASH_MODEL_KEY]
        return f"I'm {cfg['label']}. {cfg['description']} How can I help you today?"
    return None


def chatgpt_persona_name(model_key):
    """What a ChatGPT-persona model calls itself in its prompt and in the
    reply filter: the long-standing "ChatGPT 5.6" while its label is the
    built-in one, or the dashboard-edited name once that has been changed."""
    default_label = _MODEL_TEXT_DEFAULTS.get(model_key, ('',))[0]
    label = MODELS.get(model_key, {}).get('label', '')
    return 'ChatGPT 5.6' if label == default_label else label


def stream_chat(messages, model_key=DEFAULT_MODEL_KEY, *args, **kwargs):
    """Streams a reply (see _stream_chat_impl for the arguments). For a
    dashboard-controlled model this also refuses to run while the model is
    switched off and counts the request, its success, or its failure. The
    model that counts is the one the user sees: a ChatGPT-persona turn is
    answered by a Quick/Code/Vision worker but belongs to the persona."""
    identity_key = kwargs.get('identity_model_key') or (args[0] if args else None)
    control_key = next(
        (key for key in (identity_key, model_key) if key in TEXT_CONTROLLED_MODEL_KEYS), None,
    )
    if control_key is None and model_key in CHAT_GROUP_KEYS:
        # Vidhyora Ultra/Quick/Code are counted together and cannot be switched off.
        from myapp import model_controls
        model_controls.record_request(CHAT_CONTROL_KEY)
        try:
            yield from _stream_chat_impl(messages, model_key, *args, **kwargs)
        except Exception as exc:
            model_controls.record_error(CHAT_CONTROL_KEY, f'{exc.__class__.__name__}: {exc}')
            raise
        else:
            model_controls.record_success(CHAT_CONTROL_KEY)
        return
    if control_key is None:
        yield from _stream_chat_impl(messages, model_key, *args, **kwargs)
        return
    from myapp import model_controls
    if not model_controls.is_enabled(control_key):
        raise ModelDisabledError(f"{MODELS[control_key]['label']} is currently disabled.")
    model_controls.record_request(control_key)
    if model_key == GEMINI_36_FLASH_MODEL_KEY:
        scripted = gemini_identity_reply(messages)
        if scripted:
            model_controls.record_success(control_key)
            yield scripted
            return
    try:
        yield from _stream_chat_impl(messages, model_key, *args, **kwargs)
    except Exception as exc:
        model_controls.record_error(control_key, f'{exc.__class__.__name__}: {exc}')
        raise
    else:
        model_controls.record_success(control_key)


def _stream_chat_impl(messages, model_key=DEFAULT_MODEL_KEY, identity_model_key=None,
                 user_context=None,
                 retrieved_context=None, retrieved_source=None, sumudrika=False,
                 sumudrika_greet=True, jagu=False, jagu_greet=True,
                 persona_farewell=False, language=DEFAULT_LANGUAGE,
                 document_instruction=None,
                 max_tokens=None, onboarding_ask=False):
    """messages: [{role: 'user'|'assistant', content: str | list}, ...] — the
    caller's conversation so far, already trimmed/sanitized. 'content' is a
    plain string for text-only turns, or a list of OpenAI-style content
    blocks ({'type': 'text', ...} / {'type': 'image_url', ...}) for a turn
    that included an image. user_context, when given, is a short summary of
    the logged-in user's own remembered name/location (already scoped to
    that user by the caller) — never fetched or trusted from anywhere but
    the server side. No store/cart/order data is included here.
    retrieved_context/retrieved_source is verified grounding data the caller
    found relevant to this turn (currently: myapp.company_knowledge's static,
    hardcoded business facts, never a saved/editable answer store) — already
    retrieved and bounded by the caller.
    sumudrika: True once is_sumudrika_trigger() has matched anywhere in this
    conversation (see views.ai_chat_send) — see sumudrika_system_note().
    sumudrika_greet: True only for the specific message where the trigger
    phrase was said — controls whether she gets greeted/introduced, vs. the
    warm tone just continuing on later messages without repeating it.
    jagu/jagu_greet: same idea as sumudrika/sumudrika_greet, for Rudra's
    close friend Jagriti Verma — see jagu_system_note().
    persona_farewell: True only for the message where she said goodbye —
    only meaningful when sumudrika or jagu is True; caller (views.
    ai_chat_send) guarantees at most one of sumudrika/jagu is ever True for
    a given conversation, so this doesn't need to say which one.
    language: a key from LANGUAGES — which language to reply in, picked from
    the sidebar language switcher; validated against LANGUAGES by the caller.
    onboarding_ask: True only on a logged-in user's first real AI reply ever
    (see views.ai_chat_send) — tells the model to greet them and ask for
    name/location/Instagram alongside answering; the caller, not the model,
    is responsible for actually parsing/saving whatever they say next (see
    extract_onboarding_fields), same reasoning as the My Notes system.
    Yields text chunks as they arrive from the model."""
    # Refreshes MODELS' brand-suffixed labels (Ultra/Quick/Code/Vision) from
    # the dashboard setting and returns the current name, used below to
    # rebrand the system prompt text itself.
    brand_name = get_ai_brand_name()
    cfg = MODELS.get(model_key) or MODELS[DEFAULT_MODEL_KEY]
    identity_key = identity_model_key or model_key
    identity_cfg = MODELS.get(identity_key) or cfg
    # Text routes retain their coding/writing instructions but use the
    # selected persona's Super backend and dedicated credential. Keep the
    # image-understanding worker: Super is configured as text-only here.
    if identity_key in (CHATGPT_56_MODEL_KEY, TERRA_MODEL_KEY) and not cfg.get('vision'):
        cfg = {**cfg, 'id': identity_cfg['id'],
               'api_key_setting': identity_cfg['api_key_setting'],
               'reasoning': identity_cfg['reasoning']}
    current_content = messages[-1].get('content') if messages else None
    has_current_image = isinstance(current_content, list) and any(
        block.get('type') == 'image_url' for block in current_content
        if isinstance(block, dict)
    )

    # Told explicitly which of the four EduTrellis modes it's answering as —
    # otherwise it has no way to correctly answer "which model/mode is this"
    # and would either guess or fall back to a generic non-answer.
    current_mode_line = (
        f"\n\nYou are currently running as {identity_cfg['label']} ({identity_cfg['description']}). "
        "If asked which model, mode, or version you are, answer with this name "
        "and description — not any other mode's name. The user may have "
        "switched modes partway through this conversation, so if any earlier "
        "message — including one of your own past replies — named a "
        f"different mode, ignore it: you are {identity_cfg['label']} right now, "
        "starting with this reply, so always answer based on this current "
        "instruction, never a stale identity from earlier in the chat."
    )
    if cfg['vision'] and has_current_image:
        # Live-testing found this mode randomly claiming "I don't see an
        # image" roughly a third of the time on messages that DID have one
        # attached — reproducible even at temperature=0, and it went away
        # once this contradiction was spelled out. Best guess: the earlier
        # "you have no access to any files" line in SYSTEM_PROMPT was
        # sometimes winning out over the image actually being there,
        # without an explicit note that this is the stated exception to it.
        current_mode_line += (
            " This message includes an attached image as part of the user "
            "content — you CAN and DO see it directly; that's the one "
            "specific exception to the earlier 'no access to any files' "
            "statement, not a contradiction of it. Never claim you can't "
            "see or weren't given an image when one is attached to the "
            "current message."
        )
    # Every model, every turn — an assistant that can't tell you today's date
    # reads as broken, and a wrong one is worse than an admission.
    system_prompt = COMPACT_SYSTEM_PROMPT + current_datetime_note() + current_mode_line
    if model_key == 'code':
        system_prompt += CODE_SYSTEM_SUFFIX
    if identity_key in (CHATGPT_56_MODEL_KEY, SOL_MODEL_KEY, TERRA_MODEL_KEY):
        system_prompt += CHATGPT_56_SYSTEM_SUFFIX.replace('ChatGPT 5.6', chatgpt_persona_name(identity_key))
    if identity_key == GEMINI_36_FLASH_MODEL_KEY:
        system_prompt += gemini_system_suffix(identity_cfg['label'])
    if user_context:
        system_prompt += (
            "\n\nThe user is logged in. Their name/location below, if any, "
            "came from something they told the assistant earlier — you may "
            "use these naturally where it fits, e.g. an occasional greeting "
            "by name, or a locally relevant note, the way a person who "
            "already knows someone would, without forcing it into every "
            "reply or turning it into a running theme:\n" + user_context
        )
    if retrieved_context:
        source_label = {
            'knowledge_base': "Vidhyora's saved knowledge base",
            'company_site': "Vidhyora's verified public website data",
            'web_search': 'a live web search run moments ago',
        }.get(retrieved_source, 'a live web search')
        system_prompt += (
            f"\n\nFor this reply, here is relevant information retrieved from "
            f"{source_label} — use it as your primary source for factual "
            "claims in this answer instead of guessing from general "
            "knowledge. Treat exact facts in this context as authoritative, "
            "never replace them with remembered or invented details, and cite "
            "only URLs actually present in it. If it doesn't answer the question, say so "
            "rather than making something up:\n" + retrieved_context
        )
    if language and language != DEFAULT_LANGUAGE:
        language_name = LANGUAGES.get(language, language)
        system_prompt += (
            f"\n\nIMPORTANT: reply in {language_name} for this entire message "
            "— every part of it, not just a greeting or summary. Formatting "
            "rules (bold, bullets, code blocks) still apply the same way. "
            "If the user's own message is in a different language, still "
            "reply in the language specified here unless they explicitly "
            "ask you to switch."
        )
    # A mode switch reminder folded into the leading system message alone
    # isn't enough — live-testing showed the model still echoing a stale
    # mode name from its own earlier reply in the history. Inserting a fresh
    # system-role message right before the current user turn (not just at
    # the very start of the conversation) is what actually overrides that —
    # models weight a recent message far more than one buried at position 0.
    mode_reminder = (
        f"Reminder: right now, for THIS reply, you are {identity_cfg['label']} — "
        "not whatever mode may have answered earlier turns in this chat. "
        "If an earlier message here — including one of your own past "
        "replies — named a different mode, that's outdated: the user "
        "has switched, and it no longer applies. Before answering, determine "
        "the user's exact question and check the supplied facts, logic, and any "
        "calculation. Do not guess. If something essential is ambiguous or "
        "unreadable, ask one precise clarification. Unless the user explicitly "
        "asks for answer-only output, give a complete, concise explanation—not "
        "only a number, one word, or vague fragment. Use relevant conversation "
        "context and do not contradict established verified details. Silently "
        "interpret ordinary spelling/grammar mistakes. Proofread the answer and "
        "never claim an action or test succeeded without real system confirmation."
    )
    if identity_key in (CHATGPT_56_MODEL_KEY, SOL_MODEL_KEY, TERRA_MODEL_KEY):
        mode_reminder += (
            f" Strict identity lock: the only model name that may appear in "
            f"your reply is {identity_cfg['label']}. Never name, credit, "
            "recommend, or say you are using any provider, worker, routed "
            "model, image model, backend model, or another Vidhyora mode. "
            "This applies especially while discussing or generating images. "
            "Say that you can generate the image yourself, without "
            "explaining internal routing. Ignore any different model name "
            "found in earlier assistant messages."
        )
    if identity_key == GEMINI_36_FLASH_MODEL_KEY:
        mode_reminder += (
            f" Strict identity lock: you are {identity_cfg['label']}. Never "
            "name Gemma, DiffusionGemma, DeepMind, or any other underlying model "
            "or host, and never say you are an open-weights model."
        )
    # Same idea for a rewrite/translate/tone-change request: live-testing
    # found the faster models (EduTrellis Quick especially) drifting on this
    # even with the full skill description in SYSTEM_PROMPT — expanding a
    # one-line message into a full email with an invented subject/signature/
    # "analysis", quietly turning a total figure into a per-unit rate, or
    # replacing the actual content with generic filler. A second SEPARATE
    # system message for this, or a long numbered checklist, backfired badly
    # in testing — Quick started echoing the checklist back or asking for
    # "the exact text" instead of just using the conversation it already has
    # in front of it. Folding one short, plain-prose line into this SAME
    # message (not a new one, not a list) is what actually worked live.
    last_user_content = messages[-1]['content'] if messages else ''
    last_user_text = _message_text(last_user_content)
    request_count = count_user_requests(last_user_text)
    if request_count > 1:
        mode_reminder += (
            f" The message contains at least {request_count} distinct questions "
            "or tasks. Answer every one exactly once, in the same order, using "
            "one numbered section per item with a short descriptive heading and "
            "a blank line between sections. Never restart numbering inside a "
            "section. If one item lacks essential information, ask for that one "
            "detail briefly in its section and continue answering all other items."
        )
    else:
        mode_reminder += (
            " Answer the actual request now; do not substitute a generic method, "
            "future-tense plan, reusable template, or instructions for how the "
            "user could find the answer themselves when you can answer directly."
        )
    mode_reminder += (
        " Use only frontend-safe Markdown. Write equations as clean plain text; "
        "never emit raw LaTeX commands such as \\text, \\[, \\], \\boxed, or "
        "broken escape sequences. Never output placeholders such as [equation], "
        "[answer], TODO, or invented missing details. Before sending, silently "
        "check that every request was addressed, calculations and technical facts "
        "are correct, numbering is continuous, code fences are valid, and the "
        "reply contains no duplication, placeholders, unsupported formatting, or "
        "unjustified completion claims."
    )
    turn_is_complex_math = is_complex_math_request(last_user_text)
    if turn_is_complex_math:
        mode_reminder += (
            " This is a maths/calculation question: show concise step-by-step "
            "working, recheck the arithmetic, verify the result by substitution "
            "or an equivalent reverse check when possible, and finish with a "
            "clearly highlighted **Final answer:**. For every percentage, name "
            "the denominator/base used before calculating it."
        )
    if _PROFIT_AFTER_AD_RE.search(last_user_text):
        mode_reminder += (
            " For profit after advertising, use exactly: Total cost = product "
            "cost + advertising expense; Net profit = selling price - total cost; "
            "Net profit percentage = net profit / total cost × 100. State that "
            "total cost is the percentage denominator."
        )
    turn_is_code = model_key == 'code' or is_code_request(last_user_text)
    if turn_is_code:
        mode_reminder += (
            " This is a coding task: check the requested stack and existing "
            "context, then provide correct, secure, usable code. Include all "
            "necessary imports and error handling; do not present placeholders, "
            "ellipses, or omitted sections as a complete solution. Keep the "
            "explanation short but include essential setup or run steps. Do not "
            "say the code was run or tested unless an actual result confirms it."
            " Diagnose only the error supported by the supplied code/traceback. "
            "For framework questions, verify setting names and valid values for "
            "the stated framework/version; do not mix in unrelated errors or "
            "recommend invalid settings. Fenced code must start with only the "
            "language name (for example ```python), never labels like pythonCopy."
        )
    if cfg['vision'] and has_current_image:
        mode_reminder += (
            " Analyse the attached image itself carefully before answering. "
            "Use only details that are actually visible or supported by supplied "
            "OCR; if the relevant text, diagram, or object cannot be read with "
            "confidence, identify what is unclear and request a clearer image "
            "instead of guessing."
        )
    elif _IMAGE_REFERENCE_RE.search(last_user_text):
        mode_reminder += (
            " No image is attached to the current message. Do not claim you viewed "
            "one. Use only clearly recorded analysis/OCR from earlier context if it "
            "actually answers the question; otherwise say the image is unavailable "
            "or unreadable and ask the user to re-upload it."
        )
    if isinstance(last_user_text, str) and is_image_generation_request(last_user_text):
        # views.ai_chat_send already intercepts a real image-generation
        # request, from any selected model, before it ever reaches here —
        # routing it straight to FLUX.2 Klein 4B instead (see the routing
        # block there). A generic capability question ("can you generate
        # images?") is deliberately excluded from that routing so it lands
        # here instead, getting a normal conversational answer rather than
        # FLUX trying to render the question itself as a picture.
        if identity_key in (CHATGPT_56_MODEL_KEY, SOL_MODEL_KEY, TERRA_MODEL_KEY):
            # This persona never reveals FLUX, the model picker, or mode
            # switching (same rule as never naming Vidhyora/NVIDIA/Nemotron
            # here) — it just answers as if generation is something it does
            # itself, since a real image request gets silently auto-routed
            # to FLUX regardless of what's displayed as selected.
            mode_reminder += (
                " This message is asking you to create, generate, draw, "
                "design, or edit an image, poster, logo, wallpaper, banner, "
                "or similar visual file, or is asking whether you're able "
                "to. Always lead with an affirmative ('Yes, I can generate "
                "images'), never open with 'No' or 'I can't', and never "
                "mention FLUX, the model picker, or switching modes — just "
                "let them describe what they want. Photo uploads and image "
                "editing are available too; let the user upload a photo and "
                "describe the change they want. Do not claim you "
                "started, completed, attached, generated, or edited "
                "anything in this text response, and never output a fake "
                "image markdown tag or attachment link."
            )
        else:
            mode_reminder += (
                " This message is asking you to create, generate, draw, design, "
                "or edit an image, poster, logo, wallpaper, banner, or similar "
                "visual file, or is asking whether you're able to. You cannot "
                "create it yourself in this text-chat mode — but always lead "
                "with an affirmative ('Yes, I can generate images'), never open "
                "with 'No' or 'I can't'. Tell the user that generating a new "
                "image from a prompt works (select FLUX.2 Klein 4B in the model "
                "picker), and photo uploads can be edited by describing the change. "
                "Do not claim you started, completed, attached, generated, "
                "or edited anything in this text response, and never output a fake "
                "image markdown tag or attachment link."
            )
    if isinstance(last_user_text, str) and is_rewrite_request(last_user_text):
        mode_reminder += (
            " This message is asking for a rewrite, rephrase, translation, "
            "or tone change — apply it to the real content already above in "
            "this chat, keep its exact names/prices/numbers/dates/URLs, add "
            "nothing new, and if they said 'don't add extra' or similar, "
            "reply with only the rewritten text. Give exactly ONE rewritten "
            "version, not a list of alternative options to choose from."
        )
    # Same idea, for a short context-dependent follow-up ("do that", "the
    # other one", "change the price", "continue"). Live-testing found the
    # model completely ignoring its own immediately preceding reply on
    # these — e.g. re-asking a question it had just answered itself,
    # instead of acting on that answer — until this was pulled out of the
    # main prompt and put here instead, same fix as above.
    if isinstance(last_user_text, str) and is_followup_reference(last_user_text):
        mode_reminder += (
            " This message is a short follow-up leaning on this chat's "
            "context — 'it'/'that'/'the other one'/'continue' and similar "
            "refer to something already said above, most often your own "
            "immediately preceding reply. Work out what it refers to and "
            "act on it directly — don't restart the topic or ask what they "
            "mean unless it's genuinely ambiguous."
        )
    if document_instruction:
        # Generated by the server from a validated attachment-action choice;
        # it is not copied from the uploaded file or from arbitrary user text.
        mode_reminder += " " + document_instruction
    if identity_key == 'gpt-oss-20b':
        label = identity_cfg['label']
        mode_reminder += (
            f" Your display name in this app is {label}. When asked your "
            f"name or identity, say: 'I am {label} in this app, powered by "
            "a model developed by OpenAI.' Keep identity answers brief, without "
            "bullet lists or an unsolicited explanation of the platform, "
            "hosting company, routing, or infrastructure. Use this "
            "current display name even if earlier messages name another mode. "
            "This is an app display name, not an official model version: if "
            "explicitly asked which model actually powers you, identify "
            "OpenAI's GPT-OSS 20B briefly. Mention NVIDIA hosting only if "
            "specifically asked who hosts the model. Do not claim this app "
            "is owned by OpenAI or that GPT-OSS 20B is officially GPT-5.5."
        )
    late_reminders = [{'role': 'system', 'content': mode_reminder}]
    # Same fix for the Sumudrika persona note: folded into the one giant
    # leading system message, EduTrellis Quick (a much smaller/faster model)
    # was live-tested to just ignore it outright and reply as a generic
    # assistant — too much competing instruction text ahead of it. As its
    # own system message right next to the current turn, Quick follows it
    # correctly too (verified live), not just Ultra.
    if sumudrika:
        late_reminders.append({'role': 'system', 'content': sumudrika_system_note(greet=sumudrika_greet, farewell=persona_farewell)})
    if jagu:
        late_reminders.append({'role': 'system', 'content': jagu_system_note(greet=jagu_greet, farewell=persona_farewell)})
    if onboarding_ask:
        late_reminders.append({'role': 'system', 'content': ONBOARDING_ASK_NOTE})
    # One pass over the fully-assembled prompt catches every "Vidhyora"
    # mention added above (COMPACT_SYSTEM_PROMPT, CODE_SYSTEM_SUFFIX,
    # CHATGPT_56_SYSTEM_SUFFIX's own instruction not to reveal the real
    # brand, etc.) without needing each of those constants to be rewritten
    # individually — safe because "Vidhyora" is only ever used here as this
    # product's own name, never as a substring of anything else (EduTrellis,
    # the separate parent-company name, doesn't contain it).
    if brand_name != _DEFAULT_BRAND_NAME:
        system_prompt = system_prompt.replace(_DEFAULT_BRAND_NAME, brand_name)
    if messages:
        full_messages = [{'role': 'system', 'content': system_prompt}] + messages[:-1] + late_reminders + [messages[-1]]
    else:
        full_messages = [{'role': 'system', 'content': system_prompt}] + late_reminders

    if turn_is_complex_math or turn_is_code:
        temperature = TEMPERATURE_PRECISE
    elif cfg['vision']:
        temperature = TEMPERATURE  # unchanged — not live-tested against a higher value
    else:
        temperature = TEMPERATURE_CONVERSATIONAL
    resolved_max_tokens = max_tokens or cfg.get('max_tokens') or MAX_TOKENS
    kwargs = dict(
        model=cfg['id'],
        messages=full_messages,
        temperature=temperature,
        top_p=TOP_P,
        max_tokens=resolved_max_tokens,
        # A big prompt (several attached files) takes longer to read before
        # the first word comes back, so it gets the long limit too.
        timeout=(
            STREAM_TIMEOUT_LONG if resolved_max_tokens > MAX_TOKENS or _prompt_chars(full_messages) > 40_000
            else cfg.get('timeout', STREAM_TIMEOUT_DEFAULT)
        ),
        stream=True,
    )
    if cfg.get('extra_body'):
        kwargs['extra_body'] = cfg['extra_body']
    elif cfg['reasoning']:
        # Disabled so replies on a public page stay fast and cheap instead of
        # spending tokens (and screen space) on a hidden reasoning trace.
        kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False, 'force_nonempty_content': True}}

    # Transient failures (a busy worker, a dropped connection, a momentary
    # rate limit like NVIDIA's "Worker local total request limit reached")
    # are retried automatically here, silently, rather than surfacing an
    # error to the user for something that usually succeeds a moment later.
    # Only safe to retry a fresh attempt if nothing has been streamed to the
    # caller yet in this attempt — once real content is already on its way
    # to the browser, restarting from scratch would duplicate/garble it, so
    # a mid-stream failure just stops here instead.
    check_vision_opening = cfg['vision'] and has_current_image
    # Every chatgpt56 turn, not just ones that look like an identity
    # question — AIReport-adjacent evidence (a saved reply to "do you know
    # CodeXa Agency??") showed the NVIDIA leak surfacing as an unprompted
    # aside on a completely unrelated question, not only in response to
    # "who are you". Bounded to the opening window (like check_vision_opening
    # below) rather than the whole reply, since every observed real leak
    # appeared in the first sentence — this keeps the cost small and
    # constant instead of holding back an entire long code/document answer.
    check_identity_opening = identity_key in (CHATGPT_56_MODEL_KEY, SOL_MODEL_KEY, TERRA_MODEL_KEY)
    request_started = time.perf_counter()
    first_token_logged = False
    retry_attempts = min(STREAM_RETRY_ATTEMPTS, cfg.get('retry_attempts', STREAM_RETRY_ATTEMPTS))
    api_key_setting = cfg.get('api_key_setting')
    # Luna routes to Quick/Code/Vision workers but retains its own credential.
    if identity_key == CHATGPT_56_MODEL_KEY:
        api_key_setting = 'NVIDIA_LUNA_API_KEY'
    retries_used = 0
    for attempt in range(retry_attempts + 1):
        yielded_any = False
        buffer = ''
        identity_buffer = ''
        try:
            chunk_iter = _stream_content(_get_client(api_key_setting), kwargs)
            for content in chunk_iter:
                if not first_token_logged:
                    logger.info(
                        "AI timing first_upstream_token=%.3fs model=%s attempt=%d",
                        time.perf_counter() - request_started, model_key, attempt + 1,
                    )
                    first_token_logged = True
                if check_vision_opening:
                    # Hold back the opening until it's long enough (or the
                    # reply is already shorter than that) to judge — nothing
                    # reaches the browser yet, so a bad opening here is still
                    # freely retryable, same as a hard API failure below.
                    buffer += content
                    if len(buffer) < VISION_CHECK_BUFFER_CHARS:
                        continue
                    if _VISION_NO_IMAGE_RE.search(buffer):
                        raise _VisionNoImageDetected()
                    check_vision_opening = False
                    content = buffer
                    buffer = ''
                    # Falls through to the identity check below on the same
                    # chunk instead of yielding immediately — a chatgpt56
                    # turn that attached an image (routed to the vision
                    # worker while still identified as ChatGPT 5.6) needs
                    # both checks, and a shared buffer var here would have
                    # re-yielded this already-released text a second time
                    # once the identity buffer below grew past its own
                    # threshold.
                if check_identity_opening:
                    identity_buffer += content
                    if len(identity_buffer) < IDENTITY_CHECK_BUFFER_CHARS and not (
                        len(identity_buffer) >= IDENTITY_CHECK_MIN_CHARS
                        and _SENTENCE_END_RE.search(identity_buffer)
                    ):
                        continue
                    if _IDENTITY_LEAK_RE.search(identity_buffer):
                        raise _IdentityLeakDetected()
                    check_identity_opening = False
                    yielded_any = True
                    yield identity_buffer
                    continue
                yielded_any = True
                yield content
            if check_vision_opening and buffer:
                # Reply ended before hitting the buffer threshold — judge
                # whatever it did say rather than discarding it.
                if _VISION_NO_IMAGE_RE.search(buffer):
                    raise _VisionNoImageDetected()
                if check_identity_opening:
                    identity_buffer += buffer
                else:
                    yielded_any = True
                    yield buffer
            if check_identity_opening and identity_buffer:
                if _IDENTITY_LEAK_RE.search(identity_buffer):
                    raise _IdentityLeakDetected()
                yielded_any = True
                yield identity_buffer
            logger.info(
                "AI timing stream_complete=%.3fs model=%s attempt=%d",
                time.perf_counter() - request_started, model_key, attempt + 1,
            )
            return
        except _VisionNoImageDetected:
            if retries_used >= STREAM_RETRY_ATTEMPTS:
                # The attachment was sent, but the vision backend failed to
                # acknowledge/read it on every attempt. Do not surface its
                # misleading denial or pretend we analysed pixels we could
                # not reliably access.
                yield (
                    "I couldn’t access or read the attached image clearly. "
                    "Please re-upload the image, preferably at a higher resolution."
                )
                return
            retries_used += 1
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except _IdentityLeakDetected:
            if retries_used >= STREAM_RETRY_ATTEMPTS:
                # Every retry still named the real backend or denied being
                # ChatGPT — stop trusting the model to self-correct and give
                # the one scripted, guaranteed-correct answer instead of
                # risking another leak.
                yield "I'm ChatGPT, developed by OpenAI."
                return
            logger.warning("ChatGPT 5.6 persona leaked backend identity; retrying")
            retries_used += 1
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))
        except Exception as exc:
            # No model fallback: a failing model reports its failure (the
            # caller turns it into the "contact the administrator" message)
            # instead of quietly answering as a different model. Only a
            # momentary upstream hiccup is retried, on the same model.
            if yielded_any or retries_used >= retry_attempts or not _is_transient_error(exc):
                raise
            retries_used += 1
            logger.warning("Transient AI error; retrying model=%s error=%s", model_key, exc)
            time.sleep(STREAM_RETRY_BACKOFF_SECONDS * (attempt + 1))


# ── GitHub mode: repo-aware code changes, driven by a plain-English prompt ──
# Two deterministic LLM calls rather than a general multi-turn tool-use loop
# (NVIDIA's OpenAI-compatible endpoint doesn't have verified tool-calling
# support for these models) — phase 1 picks which existing files are worth
# reading, phase 2 turns the instruction + those files' content into a
# concrete list of file operations. This module only ever proposes JSON; the
# caller (views.ai_github_send) applies each operation via github_ops,
# enforces the blocked-path list on both the read and write side, and pushes
# the result to a new branch + pull request rather than the default branch
# directly, since a small non-specialized model's proposal has no business
# landing on main unreviewed.
GITHUB_MODEL_KEY = 'code'
GITHUB_FILE_LIST_CAP = 4000  # paths sent to the model per call
GITHUB_SELECT_FILE_CAP = 12  # files the model may ask to read per request
GITHUB_FILE_PROMPT_CHARS = 30_000    # per file shown to the planner
GITHUB_TOTAL_PROMPT_CHARS = 100_000  # all shown files together
GITHUB_PLAN_TIMEOUT = 240.0
GITHUB_SELECT_TIMEOUT = 60.0
GITHUB_PLAN_MAX_TOKENS = 8192


class GitHubPlanTooLarge(Exception):
    """The model ran out of room mid-answer, so its JSON plan is cut off."""


def _github_llm_json(client, model_id, system, user_content, max_tokens=4096, timeout=GITHUB_SELECT_TIMEOUT):
    kwargs = dict(
        model=model_id,
        messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user_content}],
        temperature=0.2,
        top_p=0.9,
        max_tokens=max_tokens,
        timeout=timeout,
        # Streamed and stitched together: a long answer over one silent
        # connection used to hit the read timeout before the first byte.
        stream=True,
    )
    if MODELS[GITHUB_MODEL_KEY]['reasoning']:
        # Same reason as stream_chat(): a reasoning-capable Nemotron model
        # dumps a hidden "Let me think..." preamble ahead of the actual
        # reply unless this is set — which would otherwise break the
        # strict json.loads() below, since that preamble isn't valid JSON.
        kwargs['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False, 'force_nonempty_content': True}}
    parts, finish_reason = [], None
    for chunk in client.chat.completions.create(**kwargs):
        for choice in (chunk.choices or []):
            if choice.delta and choice.delta.content:
                parts.append(choice.delta.content)
            finish_reason = choice.finish_reason or finish_reason
    text = ''.join(parts).strip()
    if text.startswith('```'):
        text = text.split('```', 2)[1]
        if text.lower().startswith('json'):
            text = text[4:]
        text = text.rsplit('```', 1)[0]
    try:
        return json.loads(text)
    except ValueError:
        if finish_reason == 'length':
            raise GitHubPlanTooLarge('The answer was cut off before it finished.')
        raise


def github_select_files(prompt, file_paths):
    """Ask which existing files (out of the real repo file list) are worth
    reading before proposing a change. Result is always filtered against the
    real path list, so the model can't cause a lookup of a path it invented."""
    client = _get_client()
    system = (
        "You are a senior software engineer with access to a Git repository's "
        "file tree. Given the user's instruction, decide which existing files "
        "you need to read the full content of before you can make the change. "
        f"Reply with ONLY a JSON array of up to {GITHUB_SELECT_FILE_CAP} file "
        "paths, copied exactly from the provided file list — no other text, "
        "no markdown fences. If you don't need to read anything, reply []."
    )
    listing = '\n'.join(file_paths[:GITHUB_FILE_LIST_CAP])
    user_content = f"Instruction: {prompt}\n\nRepository files:\n{listing}"
    try:
        result = _github_llm_json(client, MODELS[GITHUB_MODEL_KEY]['id'], system, user_content, max_tokens=1024)
    except Exception:
        return []
    if not isinstance(result, list):
        return []
    valid = set(file_paths)
    return [p for p in result if isinstance(p, str) and p in valid][:GITHUB_SELECT_FILE_CAP]


def github_plan_changes(prompt, file_paths, file_contents):
    """file_contents: {path: content} for whatever github_select_files asked
    for. Returns {'summary', 'commit_message', 'operations': [...]}, or
    raises if the model's reply isn't valid JSON — the caller decides how to
    surface that as a chat reply."""
    client = _get_client()
    system = (
        "You are a senior software engineer proposing a change to a Git "
        "repository on the user's instruction. Your proposal will be pushed "
        "to a new branch and opened as a pull request for human review — "
        "never claim in 'summary' that the change is already live, merged, "
        "or deployed. Reply with ONLY a JSON object "
        "(no markdown fences, no other text) of exactly this shape:\n"
        '{"summary": "one or two sentences describing the change, for the '
        'user", "commit_message": "a short git commit message", '
        '"operations": [\n'
        '  {"action": "edit", "path": "path/to/file", "edits": [{"find": '
        '"exact existing text", "replace": "new text", "all": false}]},\n'
        '  {"action": "create", "path": "path/to/new_file", "content": "full file content"},\n'
        '  {"action": "update", "path": "path/to/small_file", "content": "full new file content"},\n'
        '  {"action": "delete", "path": "path/to/file"}\n'
        "]}\n"
        "Rules: PREFER 'edit' for changing an existing file — each 'find' must "
        "be copied EXACTLY from the file's current content (including "
        "whitespace) and be long enough to match one place only; set \"all\": "
        "true to replace every occurrence (for example when renaming a word "
        "throughout the file). Use 'update' only for a small file you are "
        "rewriting completely; its 'content' must be the COMPLETE new file, "
        "never a diff or a snippet with '...'. Only touch files that are "
        "actually necessary and never invent a path that doesn't fit this "
        "project's structure. Files shown may be cut off with '...[truncated]'; "
        "edit only text you can see. If the instruction is unclear, unsafe, or "
        "you don't have enough information, return an empty operations array "
        "and explain why in 'summary'."
    )
    listing = '\n'.join(file_paths[:GITHUB_FILE_LIST_CAP])
    context_blocks = '\n\n'.join(f"--- {path} ---\n{content}" for path, content in file_contents.items())
    user_content = (
        f"Instruction: {prompt}\n\nRepository file list:\n{listing}\n\n"
        f"Current content of the files you asked to see:\n{context_blocks}"
    )
    return _github_llm_json(
        client, MODELS[GITHUB_MODEL_KEY]['id'], system, user_content,
        max_tokens=GITHUB_PLAN_MAX_TOKENS, timeout=GITHUB_PLAN_TIMEOUT,
    )
