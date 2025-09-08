import os
from sphinx.application import Sphinx
from sphinx.util.typing import ExtensionMetadata

def setup(app: Sphinx) -> ExtensionMetadata:
    # Define the json_url for our version switcher.
    json_url = f"https://{app.config.project}.readthedocs.io/zh-cn/latest/_static/switcher.json"
    # Define the version we use for matching in the version switcher.
    version_match = os.environ.get("READTHEDOCS_VERSION")
    # If READTHEDOCS_VERSION doesn't exist, we're not on RTD
    # If it is an integer, we're in a PR build and the version isn't correct.
    # If it's "latest" → change to "dev" (that's what we want the switcher to call it)
    if not version_match or version_match.isdigit() or version_match == "latest":
        # For local development, infer the version to match from the package.
        if "dev" in app.config.release or "rc" in app.config.release:
            version_match = "dev"
            # We want to keep the relative reference if we are in dev mode
            # but we want the whole url if we are effectively in a released version
            json_url = "_static/switcher.json"
        else:
            version_match = f"v{app.config.release}"
    elif version_match == "stable":
        version_match = f"v{app.config.release}"
    app.config["html_theme_options"].update({"switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },})
    # -- To demonstrate ReadTheDocs switcher -------------------------------------
    # This links a few JS and CSS files that mimic the environment that RTD uses
    # so that we can test RTD-like behavior. We don't need to run it on RTD and we
    # don't wanted it loaded in GitHub Actions because it messes up the lighthouse
    # results.
    if not os.environ.get("READTHEDOCS") and not os.environ.get("GITHUB_ACTIONS"):
        app.add_css_file(
            "https://assets.readthedocs.org/static/css/readthedocs-doc-embed.css"
        )
        app.add_css_file("https://assets.readthedocs.org/static/css/badge_only.css")

        # Create the dummy data file so we can link it
        # ref: https://github.com/readthedocs/readthedocs.org/blob/bc3e147770e5740314a8e8c33fec5d111c850498/readthedocs/core/static-src/core/js/doc-embed/footer.js  # noqa: E501
        app.add_js_file("rtd-data.js")
        app.add_js_file(
            "https://assets.readthedocs.org/static/javascript/readthedocs-doc-embed.js",
            priority=501,
        )
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
