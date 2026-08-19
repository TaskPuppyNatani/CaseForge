"""Small, centralized visual identity values for the CaseForge GUI."""

BRANDING_GREEN = "#4CAF50"
BRANDING_GREEN_HOVER = "#45A049"
BRANDING_PURPLE = "#B48CFF"


def taskpuppy_brand_html() -> str:
    """Return the two-color TaskPuppyKreations wordmark markup."""

    return (
        f'<span style="color: {BRANDING_PURPLE};">TaskPuppy</span>'
        f'<span style="color: {BRANDING_GREEN};">Kreations</span>'
    )
