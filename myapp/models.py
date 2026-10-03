from datetime import timedelta
import hashlib
import secrets
import uuid

from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import F
from django.utils import timezone




class StoreProfile(models.Model):
    """Extra store-specific fields for a Django auth User (E-Store signups)."""
    LOCATION_UNKNOWN = 'unknown'
    LOCATION_GRANTED = 'granted'
    LOCATION_DENIED = 'denied'
    LOCATION_CONSENT_CHOICES = [
        (LOCATION_UNKNOWN, 'Not asked'),
        (LOCATION_GRANTED, 'Enabled'),
        (LOCATION_DENIED, 'Declined'),
    ]

    user           = models.OneToOneField(User, on_delete=models.CASCADE, related_name='store_profile')
    phone          = models.CharField(max_length=20, blank=True)
    age            = models.PositiveSmallIntegerField(null=True, blank=True)
    state          = models.CharField(max_length=100, blank=True)
    city           = models.CharField(max_length=100, blank=True)
    avatar         = models.ImageField(upload_to='avatars/', blank=True, null=True)
    wallet_balance = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    manual_amount_paid = models.DecimalField(
        max_digits=12, decimal_places=2, default=0,
        help_text='Amount manually recorded as paid when staff creates or edits this customer.',
    )
    manual_payment_received_at = models.DateTimeField(
        null=True, blank=True,
        help_text='Date and time the manually recorded payment was received.',
    )
    email_verified = models.BooleanField(default=False)  # unused — verification moved to phone/SMS, see phone_verified
    phone_verified = models.BooleanField(default=False)
    location_consent = models.CharField(
        max_length=10, choices=LOCATION_CONSENT_CHOICES, default=LOCATION_UNKNOWN,
        help_text='Whether the user allowed a one-time browser location request.',
    )
    location_latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    location_longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    location_accuracy_m = models.PositiveIntegerField(null=True, blank=True)
    location_updated_at = models.DateTimeField(null=True, blank=True)
    location_place_name = models.CharField(
        max_length=255, blank=True,
        help_text='Reverse-geocoded "City, State" for location_latitude/longitude, resolved once when captured.',
    )

    # Vidhyora AI subscription — buying the AI plan product (see
    # AI_SUBSCRIPTION_PRODUCT_SLUG below) extends this instead of granting
    # a boolean flag, so back-to-back renewals stack cleanly. Staff accounts
    # bypass both the cap and this field entirely (see views._ai_profile_gate)
    # rather than being modeled as a permanent subscription here.
    ai_subscription_until = models.DateTimeField(null=True, blank=True, help_text="Vidhyora AI access is unlimited until this time. Blank/past = free tier.")
    # Set alongside ai_subscription_until at the same two grant points
    # (dashboard AddUserForm account creation, dashboard_ai_grant) and
    # cleared with it on revoke — lets the account's own Subscription tab
    # show a start date next to the expiry, not just "active until X".
    ai_subscription_started_at = models.DateTimeField(null=True, blank=True, help_text="When the current AI subscription period began.")
    ai_free_messages_used = models.PositiveIntegerField(default=0, help_text="Free-tier Vidhyora AI messages sent so far (resets on each new subscription purchase).")
    login_count = models.PositiveIntegerField(
        default=0,
        help_text='Number of successful account logins recorded by Vidhyora.',
    )

    # First-time AI chat onboarding — asked once, on this account's first
    # real AI reply, then captured from whatever they say next (see
    # views.ai_chat_send and ai_chat.extract_onboarding_fields). Values are
    # saved only if the user actually volunteers them; never required.
    ai_onboarded = models.BooleanField(default=False, help_text="Already asked the first-time name/location/Instagram question (or the account predates this feature) — never ask again.")
    ai_onboarding_pending = models.BooleanField(default=False, help_text="The question was just asked; the user's next message will be parsed for an answer.")
    ai_display_name = models.CharField(max_length=100, blank=True, help_text="Name the user gave the AI chat, if any.")
    ai_location = models.CharField(max_length=150, blank=True, help_text="Location the user gave the AI chat, if any.")
    ai_instagram_handle = models.CharField(max_length=60, blank=True, help_text="Instagram handle the user gave the AI chat, if any (no leading @).")

    class Meta:
        verbose_name = 'Store Customer Profile'
        verbose_name_plural = 'Store Customer Profiles'

    def __str__(self):
        return f"{self.user.get_full_name() or self.user.username} ({self.phone})"

    def save(self, *args, **kwargs):
        # A manual receipt must always have a reportable timestamp.  Callers
        # can supply an earlier time; otherwise the moment it is first saved
        # as paid is used.
        if self.manual_amount_paid > 0 and self.manual_payment_received_at is None:
            self.manual_payment_received_at = timezone.now()
            update_fields = kwargs.get('update_fields')
            if update_fields is not None:
                kwargs['update_fields'] = set(update_fields) | {'manual_payment_received_at'}
        elif self.manual_amount_paid <= 0:
            self.manual_payment_received_at = None
            update_fields = kwargs.get('update_fields')
            if update_fields is not None:
                kwargs['update_fields'] = set(update_fields) | {'manual_payment_received_at'}
        super().save(*args, **kwargs)

    @property
    def is_ai_subscribed(self):
        return bool(self.ai_subscription_until and self.ai_subscription_until > timezone.now())


class ActiveUserSession(models.Model):
    """The one browser session currently allowed to use a user account."""

    user = models.OneToOneField(
        User, on_delete=models.CASCADE, related_name='active_login_session',
    )
    session_key = models.CharField(max_length=40, unique=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Active User Session'
        verbose_name_plural = 'Active User Sessions'

    def __str__(self):
        return f'{self.user.username} — {self.session_key}'













class PaymentSettings(models.Model):
    """Singleton Razorpay configuration, managed from the store dashboard."""
    razorpay_key_id     = models.CharField(max_length=100, blank=True)
    razorpay_key_secret = models.CharField(max_length=100, blank=True)
    is_razorpay_enabled = models.BooleanField(default=False)
    is_test_mode        = models.BooleanField(default=True)
    cod_enabled         = models.BooleanField(default=True)
    updated_at          = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Payment Settings'
        verbose_name_plural = 'Payment Settings'

    def __str__(self):
        return 'Payment settings'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    @property
    def razorpay_ready(self):
        return bool(self.is_razorpay_enabled and self.razorpay_key_id and self.razorpay_key_secret)


class EmailSettings(models.Model):
    """Singleton SMTP configuration, managed from the store dashboard. This is
    the only source of SMTP credentials for every email the app sends (order
    confirmations, contact leads) — there is no fallback in settings.py. If
    left disabled or incomplete, no email is sent."""
    is_enabled     = models.BooleanField(default=False, help_text='Turn on to send emails using the SMTP details below. If off (or incomplete), no email is sent.')
    smtp_host      = models.CharField(max_length=200, blank=True, default='smtp.gmail.com')
    smtp_port      = models.PositiveIntegerField(default=587)
    smtp_username  = models.CharField(max_length=200, blank=True)
    smtp_password  = models.CharField(max_length=200, blank=True)
    use_tls        = models.BooleanField(default=True)
    use_ssl        = models.BooleanField(default=False)
    from_email     = models.CharField(max_length=200, blank=True, help_text='e.g. "EduTrellis <support@edutrellis.in>". Defaults to the SMTP username if left blank.')
    notify_email   = models.EmailField(blank=True, help_text='Where new-order and contact-lead notifications are sent. Defaults to the support email if left blank.')
    updated_at     = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Email (SMTP) Settings'
        verbose_name_plural = 'Email (SMTP) Settings'

    def __str__(self):
        return 'Email settings'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    @property
    def ready(self):
        return bool(self.is_enabled and self.smtp_host and self.smtp_username and self.smtp_password)






# Product id (matches Product.slug) whose first delivered order
# triggers the ₹100 wallet-credit welcome offer.
WALLET_OFFER_PRODUCT_ID = 'aud-metal'
WALLET_OFFER_CREDIT = 100

# Product id (matches Product.slug) for the Vidhyora AI monthly plan —
# seeded automatically by migration 0034_seed_ai_subscription_product.
# Buying it (see Order.maybe_grant_ai_subscription) extends the buyer's
# StoreProfile.ai_subscription_until by this many days.
AI_SUBSCRIPTION_PRODUCT_SLUG = 'edutrellis-ai-monthly'
AI_SUBSCRIPTION_DAYS = 30


class Order(models.Model):
    """A placed order, created from the cart at checkout. Product data is
    snapshotted onto OrderItem the same way CartItem snapshots it, since the
    catalogue lives in the template, not the database."""
    STATUS_PLACED     = 'placed'
    STATUS_PROCESSING = 'processing'
    STATUS_SHIPPED    = 'shipped'
    STATUS_DELIVERED  = 'delivered'
    STATUS_CANCELLED  = 'cancelled'
    STATUS_CHOICES = [
        (STATUS_PLACED, 'Placed'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_SHIPPED, 'Shipped'),
        (STATUS_DELIVERED, 'Delivered'),
        (STATUS_CANCELLED, 'Cancelled'),
    ]

    user                   = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    status                 = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PLACED)
    subtotal               = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    wallet_discount        = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    shipping_fee           = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    handling_fee           = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    total                  = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    wallet_credit_applied  = models.BooleanField(default=False)
    ai_subscription_granted = models.BooleanField(default=False)

    # Delivery address — snapshotted at checkout the same way OrderItem
    # snapshots product data, so a later profile edit never changes where an
    # already-placed order was meant to ship.
    recipient_name  = models.CharField(max_length=120, blank=True)
    recipient_phone = models.CharField(max_length=20, blank=True)
    address_line1   = models.CharField(max_length=200, blank=True)
    address_line2   = models.CharField(max_length=200, blank=True)
    city            = models.CharField(max_length=100, blank=True)
    state           = models.CharField(max_length=100, blank=True)
    pincode         = models.CharField(max_length=10, blank=True)

    created_at             = models.DateTimeField(auto_now_add=True)
    updated_at             = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Store Order'
        verbose_name_plural = 'Store Orders'

    def __str__(self):
        return f"Order #{self.pk} — {self.user.username} ({self.get_status_display()})"

    @property
    def full_address(self):
        lines = [self.address_line1, self.address_line2, self.city, self.state, self.pincode]
        return ', '.join(line for line in lines if line)

    @property
    def latest_payment(self):
        """The most recent Payment attempt for this order (COD or Razorpay,
        whatever its status). payments is ordered newest-first
        (Payment.Meta.ordering), and when the caller has
        .prefetch_related('payments') this hits that cache instead of a
        fresh query per order."""
        return self.payments.first()

    @property
    def payment_label(self):
        """'COD — pay on delivery', 'COD — Paid', 'Online — Paid',
        'Online — Pending', 'Online — Failed', etc. — shown in the
        dashboard Orders/Delivery pages so staff can see COD vs Online at
        a glance, and whether it's actually been paid."""
        payment = self.latest_payment
        if not payment:
            return None
        if payment.method == payment.METHOD_COD:
            if payment.status == payment.STATUS_COD_PENDING:
                return payment.get_status_display()
            return f'COD — {payment.get_status_display()}'
        return f'Online — {payment.get_status_display()}'

    def maybe_credit_wallet(self):
        """Credits the ₹100 welcome offer once this order is Delivered, if
        it's the customer's first order and contains the Metal Bluetooth
        Speaker. Idempotent via wallet_credit_applied — guarded with an
        atomic compare-and-swap UPDATE so two concurrent calls (e.g. a
        double-click on "Mark Delivered") can't both pass the check and
        double-credit the wallet."""
        if self.wallet_credit_applied or self.status != self.STATUS_DELIVERED:
            return
        claimed = Order.objects.filter(pk=self.pk, wallet_credit_applied=False).update(wallet_credit_applied=True)
        if not claimed:
            return
        self.wallet_credit_applied = True
        is_first_order = not Order.objects.filter(user=self.user).exclude(pk=self.pk).exists()
        has_offer_product = self.items.filter(product_id=WALLET_OFFER_PRODUCT_ID).exists()
        if is_first_order and has_offer_product:
            StoreProfile.objects.filter(user=self.user).update(wallet_balance=F('wallet_balance') + WALLET_OFFER_CREDIT)

    def maybe_grant_ai_subscription(self):
        """Extends the buyer's Vidhyora AI subscription once this order's
        AI-plan item (AI_SUBSCRIPTION_PRODUCT_SLUG) is actually paid for —
        either an online payment that's cleared, or a COD order the admin
        has marked Delivered (COD's own "paid" signal elsewhere in this
        codebase, e.g. maybe_credit_wallet above). Extends from whichever is
        later, now or the current expiry, so renewing early stacks instead
        of wasting remaining days. Idempotent the same way as
        maybe_credit_wallet — an atomic compare-and-swap on
        ai_subscription_granted means a re-saved order status or a repeated
        webhook call can't extend the subscription twice for one order."""
        if self.ai_subscription_granted:
            return
        if not self.items.filter(product_id=AI_SUBSCRIPTION_PRODUCT_SLUG).exists():
            return
        payment = self.latest_payment
        if not payment:
            return
        paid = payment.status == Payment.STATUS_PAID or (
            payment.method == Payment.METHOD_COD and self.status == self.STATUS_DELIVERED
        )
        if not paid:
            return
        claimed = Order.objects.filter(pk=self.pk, ai_subscription_granted=False).update(ai_subscription_granted=True)
        if not claimed:
            return
        self.ai_subscription_granted = True
        profile, _ = StoreProfile.objects.get_or_create(user=self.user)
        now = timezone.now()
        base = profile.ai_subscription_until if profile.ai_subscription_until and profile.ai_subscription_until > now else now
        profile.ai_subscription_until = base + timedelta(days=AI_SUBSCRIPTION_DAYS)
        profile.ai_free_messages_used = 0
        profile.save(update_fields=['ai_subscription_until', 'ai_free_messages_used'])


class OrderItem(models.Model):
    order        = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product_id   = models.CharField(max_length=40)
    product_name = models.CharField(max_length=200)
    price        = models.DecimalField(max_digits=10, decimal_places=2)
    quantity     = models.PositiveIntegerField(default=1)

    class Meta:
        verbose_name = 'Order Item'
        verbose_name_plural = 'Order Items'

    def __str__(self):
        return f"{self.product_name} x{self.quantity}"

    @property
    def subtotal(self):
        return self.price * self.quantity


class Payment(models.Model):
    """A payment attempt/record for an Order — either Cash on Delivery or a
    Razorpay transaction. One Order can have multiple Payment rows if a
    Razorpay attempt fails and the shopper retries."""
    METHOD_COD      = 'cod'
    METHOD_RAZORPAY = 'razorpay'
    METHOD_CHOICES = [
        (METHOD_COD, 'Cash on Delivery'),
        (METHOD_RAZORPAY, 'Razorpay'),
    ]

    STATUS_PENDING     = 'pending'
    STATUS_PAID        = 'paid'
    STATUS_FAILED      = 'failed'
    STATUS_REFUNDED    = 'refunded'
    STATUS_COD_PENDING = 'cod_pending'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PAID, 'Paid'),
        (STATUS_FAILED, 'Failed'),
        (STATUS_REFUNDED, 'Refunded'),
        (STATUS_COD_PENDING, 'COD — pay on delivery'),
    ]

    order                = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='payments')
    method               = models.CharField(max_length=20, choices=METHOD_CHOICES, default=METHOD_COD)
    status               = models.CharField(max_length=20, choices=STATUS_CHOICES, default=STATUS_PENDING)
    amount               = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    razorpay_order_id    = models.CharField(max_length=80, blank=True)
    razorpay_payment_id  = models.CharField(max_length=80, blank=True)
    razorpay_signature   = models.CharField(max_length=200, blank=True)
    created_at           = models.DateTimeField(auto_now_add=True)
    updated_at           = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Payment'
        verbose_name_plural = 'Payments'

    def __str__(self):
        return f"Payment for Order #{self.order_id} — {self.get_method_display()} ({self.get_status_display()})"


class DropboxSettings(models.Model):
    """Singleton Dropbox App credentials used to back up/restore db.sqlite3,
    managed from the store dashboard. Falls back to the server-level
    DROPBOX_APP_KEY/DROPBOX_APP_SECRET/DROPBOX_REFRESH_TOKEN settings (the
    same credentials dropbox_images.py already uses for AI image archiving)
    whenever a field here is left blank, so a server that already has those
    configured shows as connected without also requiring them to be
    re-entered through this dashboard form."""
    app_key       = models.CharField(max_length=200, blank=True)
    app_secret    = models.CharField(max_length=200, blank=True)
    refresh_token = models.CharField(max_length=400, blank=True)
    updated_at    = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Dropbox Backup Settings'
        verbose_name_plural = 'Dropbox Backup Settings'

    def __str__(self):
        return 'Dropbox backup settings'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    @property
    def effective_app_key(self):
        return self.app_key or getattr(settings, 'DROPBOX_APP_KEY', '')

    @property
    def effective_app_secret(self):
        return self.app_secret or getattr(settings, 'DROPBOX_APP_SECRET', '')

    @property
    def effective_refresh_token(self):
        return self.refresh_token or getattr(settings, 'DROPBOX_REFRESH_TOKEN', '')

    @property
    def is_configured(self):
        return bool(self.effective_app_key and self.effective_app_secret and self.effective_refresh_token)


class PWASettings(models.Model):
    """Singleton PWA (Progressive Web App) configuration, managed from the
    store dashboard. When enabled (and an icon is set), the storefront
    exposes a manifest + service worker and shows an 'Install App' button
    to shoppers on supporting browsers."""
    is_enabled        = models.BooleanField(default=False, help_text="Show the 'Install App' option on the homepage. Uses the default Vidhyora icon when no custom icon is uploaded.")
    app_name          = models.CharField(max_length=100, default='EduTrellis Store', help_text='Full name shown during install and on the splash screen.')
    short_name        = models.CharField(max_length=40, default='EduTrellis', help_text='Short name shown under the home-screen icon.')
    description       = models.CharField(max_length=200, blank=True, default="Shop gadgets from EduTrellis — audio, wearables, charging and more.")
    icon              = models.ImageField(upload_to='pwa/', blank=True, null=True, help_text='Square logo, ideally 512×512px or larger — used as the installed app icon.')
    theme_color       = models.CharField(max_length=7, default='#e8001e', help_text='Hex color, e.g. #e8001e — used for the browser/app toolbar.')
    background_color  = models.CharField(max_length=7, default='#ffffff', help_text='Hex color shown behind the splash screen while the app loads.')
    updated_at        = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'PWA (Install App) Settings'
        verbose_name_plural = 'PWA (Install App) Settings'

    def __str__(self):
        return 'PWA settings'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    @property
    def ready(self):
        # ai_manifest already supplies bundled 192px/512px fallback icons.
        # Requiring a custom upload here prevented an otherwise valid PWA
        # from ever reaching the homepage.
        return bool(self.is_enabled)




class EmailVerification(models.Model):
    """A pending email-verification OTP for an already-logged-in store user.
    Signup itself is never blocked on this — the account exists and is
    usable regardless of whether/when the shopper verifies. Sending the
    email is always best-effort from the caller's side."""
    user          = models.OneToOneField(User, on_delete=models.CASCADE, related_name='email_verification')
    otp           = models.CharField(max_length=6)
    attempts      = models.PositiveSmallIntegerField(default=0)
    created_at    = models.DateTimeField(auto_now_add=True)
    last_sent_at  = models.DateTimeField(auto_now_add=True)
    expires_at    = models.DateTimeField()

    class Meta:
        verbose_name = 'Pending Email Verification'
        verbose_name_plural = 'Pending Email Verifications'

    def __str__(self):
        return f"{self.user.email} (expires {timezone.localtime(self.expires_at):%d %b %H:%M})"

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at


class PhoneVerification(models.Model):
    """A pending phone-verification OTP, sent and checked via the 2Factor
    SMS API — the actual OTP digits live at 2Factor against `session_id`,
    we never generate or store them ourselves. Signup itself is never
    blocked on this — the account exists and is usable regardless of
    whether/when the shopper verifies."""
    user          = models.OneToOneField(User, on_delete=models.CASCADE, related_name='phone_verification')
    session_id    = models.CharField(max_length=100)
    phone         = models.CharField(max_length=20)
    attempts      = models.PositiveSmallIntegerField(default=0)
    created_at    = models.DateTimeField(auto_now_add=True)
    last_sent_at  = models.DateTimeField(auto_now_add=True)
    expires_at    = models.DateTimeField()

    class Meta:
        verbose_name = 'Pending Phone Verification'
        verbose_name_plural = 'Pending Phone Verifications'

    def __str__(self):
        return f"{self.phone} (expires {timezone.localtime(self.expires_at):%d %b %H:%M})"

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at




class AIConversation(models.Model):
    """One saved chat thread on /AI/, ChatGPT-style. A guest (not logged in)
    can chat for a few messages before being asked to log in — their
    conversation is kept here tied to session_key (user is null) and gets
    handed over to their account (user set, session_key cleared) the moment
    they log in or sign up, the same way an anonymous cart is merged in."""
    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_conversations', null=True, blank=True)
    session_key = models.CharField(max_length=40, blank=True, db_index=True)
    title       = models.CharField(max_length=80, blank=True)
    # Captured once at creation from the same IP-detection the chat rate
    # limiter already uses — lets staff (see dashboard AI Activity) spot a
    # repeat spammer across guest sessions/accounts sharing one connection,
    # not just within one rate-limit window. Null when the request's IP
    # couldn't be determined at all (never set to the literal 'unknown').
    ip_address  = models.GenericIPAddressField(null=True, blank=True, db_index=True)
    created_at  = models.DateTimeField(auto_now_add=True)
    updated_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'AI Conversation'
        verbose_name_plural = 'AI Conversations'

    def __str__(self):
        return self.title or f"Conversation #{self.pk}"


class AIBlock(models.Model):
    """Staff-issued block against a repeat spammer on /AI/, checked on every
    ai_chat_send call before anything else runs. Blocks by IP (works
    against a guest, and stops a logged-out spammer from just signing up
    again from the same connection) and/or by account (still works if their
    IP changes) — either or both can be set; see dashboard AI Activity for
    where these get created."""
    ip_address = models.GenericIPAddressField(null=True, blank=True, db_index=True)
    user       = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='ai_blocks')
    reason     = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='+')

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'AI Block'
        verbose_name_plural = 'AI Blocks'
        constraints = [
            models.CheckConstraint(
                condition=models.Q(ip_address__isnull=False) | models.Q(user__isnull=False),
                name='aiblock_ip_or_user_required',
            ),
        ]

    def __str__(self):
        if self.user_id:
            return f'Blocked account: {self.user.email or self.user.username}'
        return f'Blocked IP: {self.ip_address}'


class GitHubConnection(models.Model):
    """A user's personal-access-token connection to GitHub, used by
    /AI/'s GitHub mode to read files from and push commits to a chosen repo
    on their instruction. One per user, isolated by the user relation, and
    the token itself is never sent back to the browser once saved."""
    user            = models.OneToOneField(User, on_delete=models.CASCADE, related_name='github_connection')
    access_token    = models.CharField(max_length=255)
    github_username = models.CharField(max_length=120, blank=True)
    repo_full_name  = models.CharField(max_length=200, blank=True, help_text="owner/repo this connection reads/writes, e.g. 'boosternotes/EduTrellis'.")
    default_branch  = models.CharField(max_length=100, blank=True, default='main')
    connected_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'GitHub Connection'
        verbose_name_plural = 'GitHub Connections'

    def __str__(self):
        return f"{self.github_username or self.user.username} → {self.repo_full_name or '(no repo set)'}"


class KnowledgeEntry(models.Model):
    """Legacy data from the removed EduTrellis Light model and its
    saved-answer retrieval system (light_mode.py, deleted). No AI reply path
    reads from or writes to this table any more — every response is
    generated fresh, never retrieved from a saved/cached answer. Kept only
    as an inert historical record, browsable from the admin; safe to prune
    or drop entirely if it's no longer wanted."""
    SOURCE_MANUAL = 'manual'
    SOURCE_WEB = 'web_search'
    SOURCE_CHAT = 'chat'
    SOURCE_CHOICES = [(SOURCE_MANUAL, 'Manual'), (SOURCE_WEB, 'Web search'), (SOURCE_CHAT, 'Chat')]

    topic       = models.CharField(max_length=200, help_text="Short label/question this answers, e.g. 'refund policy' or 'GST registration steps'.")
    content     = models.TextField(help_text='Saved text from the retired EduTrellis Light feature — no longer used by any AI reply.')
    source      = models.CharField(max_length=20, choices=SOURCE_CHOICES, default=SOURCE_MANUAL)
    source_url  = models.URLField(blank=True, help_text='Where this was found, if saved from a web search.')
    user        = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='knowledge_entries', help_text='Set only for an entry private to one logged-in person (from a file/image upload, or their own account details). Blank = visible to everyone.')
    session_key = models.CharField(max_length=40, blank=True, db_index=True, help_text='Same idea as user, for a private entry saved from a guest (not logged in) session.')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Knowledge Entry'
        verbose_name_plural = 'Knowledge Entries (legacy — unused)'

    def __str__(self):
        return self.topic


class YouTubeDownloadJob(models.Model):
    STATUS_PENDING = 'pending'
    STATUS_WORKING = 'working'
    STATUS_READY = 'ready'
    STATUS_FAILED = 'failed'
    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'), (STATUS_WORKING, 'Working'),
        (STATUS_READY, 'Ready'), (STATUS_FAILED, 'Failed'),
    ]

    token = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='youtube_downloads')
    source_url = models.URLField(max_length=500)
    quality = models.CharField(max_length=10, default='1080')
    title = models.CharField(max_length=300, blank=True)
    status = models.CharField(max_length=12, choices=STATUS_CHOICES, default=STATUS_PENDING)
    progress = models.PositiveSmallIntegerField(default=0)
    video_path = models.CharField(max_length=500, blank=True)
    audio_path = models.CharField(max_length=500, blank=True)
    error = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    class Meta:
        ordering = ['-created_at']


class AIGeneratedFile(models.Model):
    """A text/code file created from an AI chat request.

    The opaque token is used in the download URL, while the user/session
    ownership check in the download view keeps the file private.
    """
    token       = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    user        = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_generated_files', null=True, blank=True)
    session_key = models.CharField(max_length=40, blank=True, db_index=True)
    file_name   = models.CharField(max_length=120)
    content     = models.TextField()
    created_at  = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'AI Generated File'
        verbose_name_plural = 'AI Generated Files'

    def __str__(self):
        return self.file_name


class AIMessage(models.Model):
    ROLE_USER = 'user'
    ROLE_ASSISTANT = 'assistant'
    ROLE_CHOICES = [(ROLE_USER, 'User'), (ROLE_ASSISTANT, 'Assistant')]

    conversation = models.ForeignKey(AIConversation, on_delete=models.CASCADE, related_name='messages')
    role         = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content      = models.TextField()
    image_data   = models.TextField(blank=True)  # data: URI of an attached image, if any (user turns only)
    document_name = models.CharField(max_length=255, blank=True)
    # Extracted text (capped by doc_extract.MAX_CHARS), persisted so
    # follow-up questions about the same document work without re-uploading
    # it — replayed on every turn within AI_CHAT_MAX_HISTORY, same tradeoff
    # as replaying an attached image.
    document_text = models.TextField(blank=True)
    model_key    = models.CharField(max_length=20, blank=True)  # which EduTrellis model answered (assistant turns only)
    # Comma-separated slugs of real EduTrellis Store products shown as cards
    # under this reply (assistant turns only) — see myapp.product_search.
    # Never AI-generated text; always resolved from the real Product table
    # so history replay shows the same real cards, not anything the model
    # claimed.
    product_slugs = models.CharField(max_length=250, blank=True)
    # Set when the user retries a reply or edits a sent message: that turn and
    # everything after it is replaced by the new one. It is hidden from the
    # chat and from the model's context but kept, so the admin AI Activity
    # page still shows what was actually said.
    superseded   = models.BooleanField(default=False)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        verbose_name = 'AI Message'
        verbose_name_plural = 'AI Messages'

    def __str__(self):
        return f"{self.role}: {self.content[:40]}"


class AINote(models.Model):
    """A note saved from the AI chat, Google-Keep style — created when the
    user says something like 'take this note', 'note it down', or 'save
    details' (see request_router.is_note_intent / views._ai_save_note_response).
    Owned the same dual way as AIConversation: a real account, or a guest
    tied to session_key."""
    user         = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='ai_notes')
    session_key  = models.CharField(max_length=40, blank=True, db_index=True)
    conversation = models.ForeignKey(AIConversation, on_delete=models.SET_NULL, null=True, blank=True, related_name='notes')
    heading      = models.CharField(max_length=120, blank=True)
    content      = models.TextField()
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'AI Note'
        verbose_name_plural = 'AI Notes'

    def __str__(self):
        return self.heading or f"Note #{self.pk}"


class AIUserImage(models.Model):
    """One AI-generated image, kept for the user's "My Images" gallery.

    Deliberately independent of AIConversation and AIMessage. The gallery used
    to read straight from AIMessage.image_data, which meant deleting a chat
    silently destroyed every picture generated in it — people delete old chats
    to tidy up, not to throw away their images. The conversation link below is
    SET_NULL so the image outlives the chat it came from, and nothing in the
    app deletes these rows: they go only when the account itself does.
    """
    user         = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='ai_images')
    session_key  = models.CharField(max_length=40, blank=True, db_index=True)
    conversation = models.ForeignKey(AIConversation, on_delete=models.SET_NULL, null=True, blank=True, related_name='generated_images')
    # The stored media URL produced by views._ai_flux_response — never text the
    # model wrote, and never a data: URI.
    url          = models.TextField()
    # What the user asked for, so the gallery can be searched and a picture can
    # be traced back to its request after its chat is gone.
    prompt       = models.TextField(blank=True)
    model_key    = models.CharField(max_length=20, blank=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'AI Generated Image'
        verbose_name_plural = 'AI Generated Images'
        indexes = [models.Index(fields=['user', '-created_at'])]

    def __str__(self):
        return f"Image #{self.pk}"


class AIReport(models.Model):
    """A user's 'this answer is wrong / abusive' report against one
    assistant reply, submitted from the Report button under every AI chat
    message. Snapshots the reported turn's text, visual/document evidence,
    and model at submit time (not just a message FK) so staff can still see
    what was flagged even if the message or conversation is later deleted."""
    STATUS_OPEN = 'open'
    STATUS_RESOLVED = 'resolved'
    STATUS_CHOICES = [(STATUS_OPEN, 'Open'), (STATUS_RESOLVED, 'Resolved')]

    conversation   = models.ForeignKey(AIConversation, on_delete=models.SET_NULL, null=True, blank=True, related_name='reports')
    message        = models.ForeignKey(AIMessage, on_delete=models.SET_NULL, null=True, blank=True, related_name='reports')
    # Snapshot both sides of the reported turn, including visual inputs and
    # outputs.  Reports must remain reviewable after a chat or its generated
    # media is deleted, so the message foreign key is only a convenient link,
    # never the sole copy of the evidence.
    user_prompt    = models.TextField(blank=True)
    user_image     = models.TextField(blank=True)
    user_document_name = models.CharField(max_length=255, blank=True)
    user_document_excerpt = models.TextField(blank=True)
    reported_reply = models.TextField(blank=True)
    reported_image = models.TextField(blank=True)
    model_key      = models.CharField(max_length=20, blank=True)
    explanation    = models.TextField(blank=True)
    user           = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='ai_reports')
    session_key    = models.CharField(max_length=40, blank=True, db_index=True)
    ip_address     = models.GenericIPAddressField(null=True, blank=True)
    status         = models.CharField(max_length=10, choices=STATUS_CHOICES, default=STATUS_OPEN)
    created_at     = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'AI Report'
        verbose_name_plural = 'AI Reports'

    def __str__(self):
        return f"Report #{self.pk} ({self.get_status_display()})"


class SiteCustomization(models.Model):
    """Singleton branding configuration, managed from the store dashboard's
    Customize page — the favicon plus the title, description, and image used
    when a homepage link is shared on WhatsApp/social platforms. See
    myapp.views.site_customization_context, registered as a global template
    context processor in edutrellis/settings.py, for how SITE_FAVICON_URL
    reaches every template without each view needing to fetch this itself."""
    favicon    = models.ImageField(upload_to='branding/', blank=True, null=True, help_text='Browser-tab icon. Square, ideally 512×512px or smaller (PNG/ICO). Leave blank to use the default EduTrellis favicon.')
    ai_brand_name = models.CharField(
        max_length=60, default='Vidhyora',
        help_text='The AI assistant\'s name — shown across the chat UI (header, title, model picker) and used '
                   'in its own replies ("I\'m an AI model built by the ... team"). Changing this takes effect '
                   'immediately, no restart needed.',
    )
    social_preview_title = models.CharField(
        max_length=120, default='Vidhyora AI — Free AI Chat Assistant',
        help_text='Heading shown in WhatsApp and social link previews.',
    )
    social_preview_description = models.CharField(
        max_length=300,
        default='Chat with Vidhyora AI for product help, quick answers and learning support — free, right from your browser.',
        help_text='Short description shown below the preview heading.',
    )
    social_preview_image = models.ImageField(
        upload_to='branding/social/', blank=True, null=True,
        help_text='Large preview image. 1200×630px is recommended. Leave blank to use the default cover.',
    )
    support_whatsapp_number = models.CharField(
        max_length=20, default='9695953183',
        help_text='Shown in the in-app "Contact support" popup\'s WhatsApp option, exactly as typed. Also used '
                   'to build its wa.me link — with or without a country code both work.',
    )
    support_email = models.EmailField(
        max_length=254, default='support@edutrellis.in',
        help_text='Shown in the in-app "Contact support" popup\'s email option and used for its mailto: link.',
    )
    youtube_cookies_txt = models.TextField(
        blank=True,
        help_text='Paste a YouTube cookies.txt (Netscape format), exported while logged into YouTube in a normal '
                   'browser — fixes "Sign in to confirm you\'re not a bot" download failures, which datacenter/'
                   'server IPs (like most hosting providers) hit far more than a home connection. Leave blank to '
                   'disable. Superadmin-only feature — see myapp.youtube_download.',
    )
    site_disabled = models.BooleanField(
        default=False,
        help_text='Shows a bare 404 "Site Not Found" page to every visitor instead of the real site. The '
                   'dashboard itself (and /admin/) stays reachable so staff can always log in and switch this '
                   'back off — see myapp.middleware.SiteDisabledMiddleware.',
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'Site Customization'
        verbose_name_plural = 'Site Customization'

    def __str__(self):
        return 'Site customization'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj


class AIAccountMessageSettings(models.Model):
    """Reusable message shown after staff create an AI premium account."""

    DEFAULT_TEMPLATE = (
        "✨ Your personal AI account has been successfully activated for {access_days} days! 🎉\n"
        "Enjoy access to powerful AI models, image and file uploads, and other premium features "
        "through your dedicated account. 🚀\n"
        "🔗 Login: https://www.vidhyora.online\n"
        "📧 Email: {email}\n"
        "🔑 Password: {password}\n"
        "📅 Validity: {access_days} days\n"
        "🔒 This is your private account, and no account sharing is required. Please use the service "
        "responsibly. Fair-use policies and platform limits may apply. ⚖️\n"
        "🛠️ If you face any login or technical issue, please contact us—we’re always happy to help. 🤝\n"
        "🌟 EduTrellis\n"
        "🌐 https://www.edutrellis.in\n"
        "📧 support@edutrellis.in 📞 Calling Support: 10 AM–7 PM 💬 WhatsApp Support Available"
    )

    message_template = models.TextField(default=DEFAULT_TEMPLATE)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'AI Account Message Settings'
        verbose_name_plural = 'AI Account Message Settings'

    def __str__(self):
        return 'AI account message settings'

    @classmethod
    def get_solo(cls):
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj

    def render_message(self, *, email, password, access_days):
        return (
            self.message_template
            .replace('{email}', str(email))
            .replace('{password}', str(password))
            .replace('{access_days}', str(access_days))
        )


class AIAPIAccess(models.Model):
    """Which of ai_chat.MODELS a user's developer API key (see AIAPIKey) is
    allowed to call directly over HTTP, from their own code — separate from
    StoreProfile.ai_subscription_until, which only controls in-app chat
    access on the /AI/ page. Granted per-user by staff (dashboard API
    Management); a user with no row here, or an empty model_keys, has no
    developer API access at all and the profile page tells them to contact
    an admin."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='ai_api_access')
    model_keys = models.CharField(
        max_length=500, blank=True,
        help_text='Comma-separated ai_chat.MODELS keys (e.g. "sol,terra,luna") this account\'s API key may call.',
    )
    granted_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='+',
        help_text='Staff account that last changed this grant.',
    )
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'AI API Access Grant'
        verbose_name_plural = 'AI API Access Grants'

    def __str__(self):
        return f"{self.user} -> {self.model_keys or '(none)'}"

    @property
    def model_key_list(self):
        return [key.strip() for key in self.model_keys.split(',') if key.strip()]


class AIAPIKey(models.Model):
    """One user's developer API key for calling their AIAPIAccess-granted
    models directly (see views.api_chat_completions). Only a salted hash is
    stored — the raw key is shown once, at generation time, and can never be
    retrieved again; regenerating replaces it and immediately invalidates
    the previous key."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='ai_api_key')
    key_hash = models.CharField(max_length=64, unique=True, db_index=True)
    key_prefix = models.CharField(max_length=12, help_text='First few characters, shown in the UI so the user can recognize their key.')
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = 'AI API Key'
        verbose_name_plural = 'AI API Keys'

    def __str__(self):
        return f"{self.user} ({self.key_prefix}…)"

    @staticmethod
    def _hash(raw_key):
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    @classmethod
    def generate_for(cls, user):
        """Create (or replace) this user's key and return the one-time raw
        value — nothing else ever exposes it again."""
        raw_key = 'vdk_' + secrets.token_urlsafe(32)
        key_hash = cls._hash(raw_key)
        key_prefix = raw_key[:11]
        cls.objects.update_or_create(
            user=user,
            defaults={'key_hash': key_hash, 'key_prefix': key_prefix},
        )
        return raw_key

    @classmethod
    def resolve(cls, raw_key):
        """Look up the owning AIAPIKey for a raw key presented to the API,
        or None if it doesn't match any stored hash."""
        raw_key = (raw_key or '').strip()
        if not raw_key:
            return None
        return cls.objects.select_related('user').filter(key_hash=cls._hash(raw_key)).first()


class AICodingKey(models.Model):
    """A key for the "Start coding" CLI (OpenCode pointed at /api/v1/code/).
    Separate from AIAPIKey so the developer API and the coding CLI can be
    rotated or revoked independently, and so coding access follows the in-app
    AI plan rather than a staff grant. Like AIAPIKey only a hash is stored:
    the raw ``vdc_…`` value is shown once.

    A user has at most one *manual* key (label '', made from the panel's
    "Create key" button) plus one *automatic* key per computer that ran the
    one-line setup (label = that computer's name), so signing in a second
    machine never signs out the first."""
    MAX_PER_USER = 10

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_coding_keys')
    label = models.CharField(max_length=60, blank=True, help_text="Empty for the manual key; otherwise the computer it was set up on.")
    key_hash = models.CharField(max_length=64, unique=True, db_index=True)
    key_prefix = models.CharField(max_length=12)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    request_count = models.PositiveIntegerField(default=0)

    class Meta:
        verbose_name = 'AI Coding Key'
        verbose_name_plural = 'AI Coding Keys'

    def __str__(self):
        return f"{self.user} {self.label or 'manual'} ({self.key_prefix}…)"

    @staticmethod
    def _hash(raw_key):
        return hashlib.sha256(raw_key.encode('utf-8')).hexdigest()

    @classmethod
    def generate_for(cls, user, label=''):
        """Create (or replace the same-label) key and return the one-time raw
        value; replacing immediately invalidates that label's previous key."""
        raw_key = 'vdc_' + secrets.token_urlsafe(32)
        cls.objects.update_or_create(
            user=user, label=label,
            defaults={'key_hash': cls._hash(raw_key), 'key_prefix': raw_key[:11],
                      'last_used_at': None, 'request_count': 0, 'created_at': timezone.now()},
        )
        extra = cls.objects.filter(user=user).exclude(label='').order_by('-created_at')[cls.MAX_PER_USER:]
        cls.objects.filter(pk__in=[row.pk for row in extra]).delete()
        return raw_key

    @classmethod
    def resolve(cls, raw_key):
        raw_key = (raw_key or '').strip()
        if not raw_key:
            return None
        return cls.objects.select_related('user').filter(key_hash=cls._hash(raw_key)).first()


class AICodingDeviceCode(models.Model):
    """One run of the automatic setup script waiting for the user to approve
    it in the browser (a "device login"). The script holds the secret
    device code (only its hash is stored here); the user sees the short
    user_code on the approval page and confirms it matches their terminal."""
    STATUS_PENDING = 'pending'
    STATUS_APPROVED = 'approved'
    STATUS_DENIED = 'denied'
    STATUS_USED = 'used'

    device_hash = models.CharField(max_length=64, unique=True)
    user_code = models.CharField(max_length=9, unique=True, db_index=True)
    machine = models.CharField(max_length=60, blank=True)
    requester_ip = models.CharField(max_length=64, blank=True)
    status = models.CharField(max_length=10, default=STATUS_PENDING)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='+')
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()

    class Meta:
        verbose_name = 'AI Coding Device Code'
        verbose_name_plural = 'AI Coding Device Codes'

    def __str__(self):
        return f"{self.user_code} ({self.status})"

    @property
    def is_expired(self):
        return timezone.now() >= self.expires_at


class ProviderAPICredential(models.Model):
    """Dashboard-editable override for one of the outbound provider API keys
    normally hardcoded/env-configured in edutrellis/settings.py (see
    myapp.provider_keys.get_key, which every live call site reads through
    instead of settings directly). A blank/missing row here just means "use
    the settings.py default" — nothing breaks if this table is empty."""
    setting_name = models.CharField(
        max_length=80, unique=True, db_index=True,
        help_text='The Django settings attribute this overrides, e.g. NVIDIA_LUNA_API_KEY.',
    )
    value = models.CharField(max_length=500, blank=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, blank=True, related_name='+',
    )

    class Meta:
        verbose_name = 'Provider API Credential'
        verbose_name_plural = 'Provider API Credentials'

    def __str__(self):
        return self.setting_name


class AIModelControl(models.Model):
    """Dashboard on/off switch plus live usage counters for one picker model
    (keyed by its ai_chat.MODELS key). No row means "enabled, never used" so
    nothing breaks on a fresh database. Counters track real upstream requests
    made through ai_chat.stream_chat, including ones that failed, which the
    AIMessage-based totals on the API Data page cannot see."""
    model_key = models.CharField(max_length=60, unique=True, db_index=True)
    is_enabled = models.BooleanField(default=True)
    # Blank means "use the built-in name/description from ai_chat.MODELS".
    display_name = models.CharField(max_length=60, blank=True)
    description = models.CharField(max_length=300, blank=True)
    request_count = models.PositiveIntegerField(default=0)
    success_count = models.PositiveIntegerField(default=0)
    error_count = models.PositiveIntegerField(default=0)
    last_used_at = models.DateTimeField(null=True, blank=True)
    last_error = models.CharField(max_length=300, blank=True)
    last_tested_at = models.DateTimeField(null=True, blank=True)
    last_test_ok = models.BooleanField(null=True, blank=True)
    last_test_message = models.CharField(max_length=300, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = 'AI Model Control'
        verbose_name_plural = 'AI Model Controls'

    def __str__(self):
        return f"{self.model_key} ({'on' if self.is_enabled else 'off'})"
