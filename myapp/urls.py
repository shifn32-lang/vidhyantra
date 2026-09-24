from django.urls import path
from django.views.generic import RedirectView
from myapp import views

urlpatterns = [
    path('', views.ai_page, name='home'),
    # Keep a service worker at the conventional site-root URL. This also
    # satisfies browsers that retain an older root registration.
    path('sw.js', views.pwa_service_worker, name='service_worker'),
    # Common lowercase/typo ways people type the AI page — all land on the
    # canonical root homepage. /AI/ used to render the same page a second
    # time under its own URL (a duplicate-content SEO problem — canonical/
    # og:url/JSON-LD all read request.path, so that copy self-declared as
    # canonical instead of pointing back at "/"); it now redirects too.
    path('ai/', RedirectView.as_view(url='/', permanent=False)),
    path('aii/', RedirectView.as_view(url='/', permanent=False)),
    path('AI/', RedirectView.as_view(url='/', permanent=True)),

    path('store/dashboard/', views.dashboard_home, name='dashboard_home'),
    path('store/dashboard/signups/', views.dashboard_signups, name='dashboard_signups'),
    path('store/dashboard/user-data/', views.dashboard_user_data, name='dashboard_user_data'),
    path('store/dashboard/user-data/location-backfill/start/', views.dashboard_location_backfill_start, name='dashboard_location_backfill_start'),
    path('store/dashboard/user-data/location-backfill/status/', views.dashboard_location_backfill_status, name='dashboard_location_backfill_status'),
    path('store/dashboard/users/add/', views.dashboard_user_add, name='dashboard_user_add'),
    path('store/dashboard/signups/<int:pk>/edit/', views.dashboard_signup_edit, name='dashboard_signup_edit'),
    path('store/dashboard/signups/<int:pk>/reset-password/', views.dashboard_signup_reset_password, name='dashboard_signup_reset_password'),
    path('store/dashboard/signups/<int:pk>/delete/', views.dashboard_signup_delete, name='dashboard_signup_delete'),
    path('store/dashboard/ai/', views.dashboard_ai_management, name='dashboard_ai_management'),
    path('store/dashboard/ai/message-template/', views.dashboard_ai_message_template_save, name='dashboard_ai_message_template_save'),
    path('store/dashboard/ai/grant/', views.dashboard_ai_grant, name='dashboard_ai_grant'),
    path('store/dashboard/ai/<int:pk>/revoke/', views.dashboard_ai_revoke, name='dashboard_ai_revoke'),
    path('store/dashboard/ai/activity/', views.dashboard_ai_activity, name='dashboard_ai_activity'),
    path('store/dashboard/ai/activity/<int:pk>/', views.dashboard_ai_activity_detail, name='dashboard_ai_activity_detail'),
    path('store/dashboard/ai/activity/message/<int:pk>/image/', views.dashboard_ai_message_image, name='dashboard_ai_message_image'),
    path('store/dashboard/ai/activity/block/', views.dashboard_ai_block, name='dashboard_ai_block'),
    path('store/dashboard/ai/activity/unblock/<int:pk>/', views.dashboard_ai_unblock, name='dashboard_ai_unblock'),
    path('store/dashboard/ai/reports/', views.dashboard_ai_reports, name='dashboard_ai_reports'),
    path('store/dashboard/ai/reports/<int:pk>/', views.dashboard_ai_report_detail, name='dashboard_ai_report_detail'),
    path('store/dashboard/ai/reports/<int:pk>/image/<str:side>/', views.dashboard_ai_report_image, name='dashboard_ai_report_image'),
    path('store/dashboard/ai/reports/<int:pk>/status/', views.dashboard_ai_report_status_update, name='dashboard_ai_report_status_update'),
    path('store/dashboard/ai/reports/<int:pk>/delete/', views.dashboard_ai_report_delete, name='dashboard_ai_report_delete'),
    path('store/dashboard/api-data/', views.dashboard_api_data, name='dashboard_api_data'),
    path('store/dashboard/api-management/', views.dashboard_api_management, name='dashboard_api_management'),
    path('store/dashboard/api-management/grant/', views.dashboard_api_access_grant, name='dashboard_api_access_grant'),
    path('store/dashboard/api-management/<int:pk>/revoke/', views.dashboard_api_access_revoke, name='dashboard_api_access_revoke'),
    path('store/dashboard/payment-settings/', views.dashboard_payment_settings, name='dashboard_payment_settings'),
    path('store/dashboard/email-settings/', views.dashboard_email_settings, name='dashboard_email_settings'),
    path('store/dashboard/email-settings/test/', views.dashboard_email_settings_test, name='dashboard_email_settings_test'),
    path('store/dashboard/pwa-settings/', views.dashboard_pwa_settings, name='dashboard_pwa_settings'),
    path('store/dashboard/customize/', views.dashboard_customize, name='dashboard_customize'),
    path('store/dashboard/payments/', views.dashboard_payments, name='dashboard_payments'),
    path('store/dashboard/backup/', views.dashboard_backup, name='dashboard_backup'),
    path('store/dashboard/backup/settings/', views.dashboard_backup_settings, name='dashboard_backup_settings'),
    path('store/dashboard/backup/run/', views.dashboard_backup_run, name='dashboard_backup_run'),
    path('store/dashboard/backup/restore/', views.dashboard_backup_restore, name='dashboard_backup_restore'),
    path('store/dashboard/backup/delete-all/', views.dashboard_backup_delete_all, name='dashboard_backup_delete_all'),
    path('store/dashboard/logout/', views.dashboard_logout, name='dashboard_logout'),

    path('AI/manifest.json', views.ai_manifest, name='ai_manifest'),
    path('AI/pwa-icon/<int:size>.png', views.ai_pwa_icon, name='ai_pwa_icon'),
    path('AI/sw.js', views.pwa_service_worker, name='ai_service_worker'),
    path('AI/api/signup/', views.ai_signup, name='ai_signup'),
    path('AI/api/login/', views.ai_login, name='ai_login'),
    path('AI/api/logout/', views.ai_logout, name='ai_logout'),
    path('AI/api/location/', views.ai_location_update, name='ai_location_update'),
    path('AI/api/amount-paid/', views.ai_amount_paid_update, name='ai_amount_paid_update'),
    path('AI/api/profile/update/', views.ai_profile_update, name='ai_profile_update'),
    path('AI/api/profile/password/', views.ai_password_change, name='ai_password_change'),
    path('AI/api/profile/phone/verify/send/', views.ai_phone_verify_send, name='ai_phone_verify_send'),
    path('AI/api/profile/phone/verify/confirm/', views.ai_phone_verify_confirm, name='ai_phone_verify_confirm'),
    path('AI/api/send/', views.ai_chat_send, name='ai_chat_send'),
    path('AI/api/files/<uuid:token>/download/', views.ai_generated_file_download, name='ai_generated_file_download'),
    path('AI/api/extract/', views.ai_extract_document, name='ai_extract_document'),
    path('AI/api/convert/', views.ai_convert_file, name='ai_convert_file'),
    path('AI/api/transcribe/', views.ai_transcribe_audio, name='ai_transcribe_audio'),
    path('AI/api/youtube/start/', views.ai_youtube_start, name='ai_youtube_start'),
    path('AI/api/youtube/<uuid:token>/status/', views.ai_youtube_status, name='ai_youtube_status'),
    path('AI/api/youtube/<uuid:token>/file/<str:file_kind>/', views.ai_youtube_file, name='ai_youtube_file'),
    path('AI/api/conversations/', views.ai_conversations_list, name='ai_conversations_list'),
    path('AI/api/conversations/import/', views.ai_conversation_import, name='ai_conversation_import'),
    path('AI/api/conversations/<int:conversation_id>/', views.ai_conversation_messages, name='ai_conversation_messages'),
    path('AI/api/conversations/<int:conversation_id>/delete/', views.ai_conversation_delete, name='ai_conversation_delete'),
    path('AI/api/conversations/<int:conversation_id>/export/<str:file_format>/', views.ai_conversation_export, name='ai_conversation_export'),
    path('AI/api/notes/', views.ai_notes_list, name='ai_notes_list'),
    path('AI/api/notes/<int:note_id>/delete/', views.ai_note_delete, name='ai_note_delete'),
    path('AI/api/account/', views.ai_account_details, name='ai_account_details'),
    path('AI/api/images/delete-all/', views.ai_images_delete_all, name='ai_images_delete_all'),
    path('AI/api/report/', views.ai_report_submit, name='ai_report_submit'),
    path('AI/api/developer-key/generate/', views.ai_developer_key_generate, name='ai_developer_key_generate'),

    # Public developer API — a granted user's own code calling their
    # allowed models directly, authenticated by AIAPIKey (Authorization:
    # Bearer <key>) rather than a browser session. See views.api_chat_completions.
    path('api/v1/chat/', views.api_chat_completions, name='api_chat_completions'),

    path('AI/api/github/status/', views.github_status, name='github_status'),
    path('AI/api/github/oauth/start/', views.github_oauth_start, name='github_oauth_start'),
    path('AI/api/github/oauth/callback/', views.github_oauth_callback, name='github_oauth_callback'),
    path('AI/api/github/connect/', views.github_connect, name='github_connect'),
    path('AI/api/github/repos/', views.github_repos, name='github_repos'),
    path('AI/api/github/repo/', views.github_set_repo, name='github_set_repo'),
    path('AI/api/github/disconnect/', views.github_disconnect, name='github_disconnect'),
    path('AI/api/github/send/', views.ai_github_send, name='ai_github_send'),
]
