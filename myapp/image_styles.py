"""Ready-made picture styles for the "More images" gallery in the AI chat.

Each style is a prompt that ends where the person adds their own subject
("... of " -> "a golden retriever puppy"). Styles marked ``photo`` change a
picture the person attaches instead. The preview images in
myapp/static/ai/styles/ were made with the same prompts and the sample
subject below (see make_previews), so a preview shows what the style gives.
"""

STYLES = [
    {'id': 'eighties', 'title': "'80s flashback",
     'prompt': 'Create an image of a 1980s flashback portrait — bright retro windbreaker, aviator sunglasses, big hair, warm vintage film grain, studio backdrop — of ',
     'sample': 'a young Indian man with curly hair'},
    {'id': 'good-morning', 'title': 'Good morning',
     'prompt': 'Create an image of a peaceful good morning scene with no text or lettering — sunrise over misty hills, a steaming cup of tea on a wooden table, fresh flowers, warm golden light — with ',
     'sample': 'a small notebook and a pen beside the cup'},
    {'id': 'cricket', 'title': 'Cricket victory',
     'prompt': 'Create an image of a joyful cricket victory scene in a floodlit stadium at night — cheering crowd, confetti, flags, cinematic photo — featuring ',
     'sample': 'a young fan in a blue jersey celebrating with arms raised'},
    {'id': 'aerial', 'title': 'Aerial view',
     'prompt': 'Create an image of a top-down aerial drone photo on fresh green grass, soft natural light, lots of empty space around the subject, of ',
     'sample': 'a small chihuahua puppy sitting and looking up'},
    {'id': 'collector-tin', 'title': "Collector's tin",
     'prompt': "Create an image of an open vintage collector's tin box filled with tiny keepsakes, photos and miniature objects, top-down studio photo, themed around ",
     'sample': 'a mountain hiking trip'},
    {'id': 'caricature', 'title': 'Caricature',
     'prompt': 'Create an image of a friendly cartoon caricature with a big expressive head, small body, bright colours and clean outlines, of ',
     'sample': 'a cheerful girl explorer holding a book and a paint palette'},
    {'id': 'sketch', 'title': 'Sketch',
     'prompt': 'Create an image of a simple hand-drawn black ink doodle on plain white paper with one bright colour accent, minimal and playful, of ',
     'sample': 'a daisy flower with a tiny bee'},
    {'id': 'stickers', 'title': 'Stickers',
     'prompt': 'Create an image of a set of glossy die-cut stickers with thick white borders on a plain background, cute flat illustration style, of ',
     'sample': 'a black cat, a potted plant and a smiling face'},
    {'id': 'photo-booth', 'title': 'Photo booth',
     'prompt': 'Create an image of a black-and-white photo booth picture, candid laughing poses, soft flash, film grain, of ',
     'sample': 'two best friends'},
    {'id': 'y2k', 'title': 'Y2K digicam',
     'prompt': 'Create an image of an early-2000s digital camera flash photo, Y2K fashion, slightly overexposed, glossy colours, of ',
     'sample': 'a young woman in a pink velour jacket with tinted sunglasses'},
    {'id': 'anime', 'title': 'Anime',
     'prompt': 'Create an image of an anime-style illustration with clean line art, vibrant colours and soft cel shading, of ',
     'sample': 'a smiling young man with a moustache pointing at the viewer'},
    {'id': 'seventies', 'title': "'70s portrait",
     'prompt': 'Create an image of a 1970s film portrait — warm faded colours, retro patterned shirt, palm trees, golden-hour sun — of ',
     'sample': 'a young man with long hair and sunglasses'},
    {'id': 'impressionist', 'title': 'Impressionist painting',
     'prompt': 'Create an image of an impressionist oil painting with visible brushstrokes, dappled light and soft pastel colours, of ',
     'sample': 'a woman looking at a sunlit sea from a hillside'},
    {'id': 'pins', 'title': 'Pin collection',
     'prompt': 'Create an image of a collection of shiny enamel pins arranged on a pastel board, flat lay photo, themed around ',
     'sample': 'tennis, food and travel'},
    {'id': 'underwater', 'title': 'Underwater',
     'prompt': 'Create an image of a dreamy underwater portrait — rippling light, floating hair, tiny bubbles, calm blue water — of ',
     'sample': 'a young woman with long dark hair'},
    {'id': 'claymation', 'title': 'Claymation world',
     'prompt': 'Create an image of a claymation stop-motion scene with handmade clay textures and a tiny city set at night, of ',
     'sample': 'a cheerful man in a puffer jacket with earphones'},
    {'id': 'model-kit', 'title': 'Model kit',
     'prompt': 'Create an image of a plastic model kit sprue in one bright colour with all the unassembled parts laid out, studio photo, of ',
     'sample': 'a man with a bicycle and city buildings'},
    {'id': 'arcade', 'title': '16-bit arcade',
     'prompt': 'Create an image of a 16-bit pixel-art arcade game scene with a neon city street at night, of ',
     'sample': 'a fighter girl in a red jacket and boots'},
    {'id': 'scribble', 'title': 'Scribble',
     'prompt': "Create an image of a childlike crayon and marker scribble drawing on white paper, wobbly lines, bright colours, of ",
     'sample': 'a happy couple at a party'},
    {'id': 'bobblehead', 'title': 'Bobblehead',
     'prompt': 'Create an image of a bobblehead figurine with an oversized head on a busy city crosswalk, shallow depth of field, of ',
     'sample': 'a young man with bright pink curly hair and a backpack'},
    {'id': 'mini-me', 'title': 'Mini me',
     'prompt': 'Create an image of a cosy photo of a person at a table with tiny miniature figurines of themselves playing around, tilt-shift look, of ',
     'sample': 'a smiling man with a beard holding a mug'},
    {'id': 'cross-section', 'title': 'Cross-section',
     'prompt': 'Create an image of a detailed technical cross-section cutaway illustration showing the inner parts, clean studio background, of ',
     'sample': 'a wireless earbud'},
    {'id': 'wanderlust', 'title': 'Wanderlust',
     'prompt': 'Create an image of a travel scrapbook page with polaroid photos, tickets, pressed flowers and handwritten notes about ',
     'sample': 'a summer trip to the Greek islands'},
    {'id': 'night-flash', 'title': 'Nighttime flash',
     'prompt': 'Create an image of a candid nighttime photo with harsh direct camera flash and a dark street behind, of ',
     'sample': 'a happy French bulldog'},
    {'id': 'blueprint', 'title': 'Blueprint poster',
     'prompt': 'Create an image of a blueprint-style technical drawing poster with white lines on a blue grid and measurement marks, of ',
     'sample': 'a strawberry'},
    {'id': 'comic', 'title': 'Comic',
     'prompt': 'Create an image of a bold pop-art comic panel with thick outlines, halftone dots and bright flat colours, of ',
     'sample': 'a cat napping inside a fish-shaped bed'},
    {'id': 'film-strip', 'title': 'Film strip',
     'prompt': 'Create an image of a vintage film strip with three sequential frames telling a tiny story, of ',
     'sample': 'an astronaut floating in space'},
    {'id': 'hyperreal', 'title': 'Hyperreal wallpaper',
     'prompt': 'Create an image of a hyper-realistic macro wallpaper, ultra detailed, shallow depth of field, of ',
     'sample': 'a curled fern frond covered in dew'},
    {'id': 'product', 'title': 'Product shot',
     'prompt': 'Create an image of a clean studio product photo on a soft pastel background with gentle shadows, of ',
     'sample': 'a glass perfume bottle with flowers'},
    # Change a photo the person attaches.
    {'id': 'photo-anime', 'title': 'Anime me', 'photo': True,
     'prompt': 'Change this photo into an anime-style illustration with clean line art and vibrant colours, keeping the same person, pose and background.',
     'sample': 'an anime-style illustration of a smiling young woman in a cafe'},
    {'id': 'photo-cartoon', 'title': '3D cartoon me', 'photo': True,
     'prompt': 'Change this photo into a cute 3D animated cartoon character, keeping the same person, clothes and pose.',
     'sample': 'a cute 3D animated cartoon character of a young man in a hoodie'},
    {'id': 'photo-enhance', 'title': 'Enhance photo', 'photo': True,
     'prompt': 'Enhance this photo: sharper details, better lighting and natural colours, without changing the person or the scene.',
     'sample': 'a sharp, well-lit family photo at a birthday party with balloons'},
    {'id': 'photo-painting', 'title': 'Oil painting me', 'photo': True,
     'prompt': 'Change this photo into a classic oil painting portrait with rich brushstrokes, keeping the same person.',
     'sample': 'a classic oil painting portrait of an elderly man with a kind smile'},
]


def gallery():
    """What the page needs for each card."""
    return [
        {
            'id': s['id'], 'title': s['title'], 'prompt': s['prompt'], 'photo': bool(s.get('photo')),
            'thumb': f"ai/styles/{s['id']}.webp",
        }
        for s in STYLES
    ]


def make_previews(out_dir, only=None):
    """Generate each style's preview with the image model (run once, by hand)."""
    import io
    import os
    from PIL import Image
    from myapp import image_generation

    os.makedirs(out_dir, exist_ok=True)
    for style in STYLES:
        if only and style['id'] not in only:
            continue
        prompt = style['sample'] if style.get('photo') else style['prompt'] + style['sample']
        if style.get('photo'):
            prompt = 'Create an image of ' + prompt
        image = image_generation.generate_image(prompt, size=(896, 1152))
        with Image.open(io.BytesIO(image.content)) as opened:
            picture = opened.convert('RGB').resize((360, 463), Image.Resampling.LANCZOS)
        picture.save(os.path.join(out_dir, f"{style['id']}.webp"), 'WEBP', quality=80, method=6)
        yield style['id']
