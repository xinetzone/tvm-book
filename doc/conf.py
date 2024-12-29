import os
# === Path setup =====================================================================================
import sys
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT/'doc')])
from utils.links import icon_links

# == Project 信息 =====================================================================================
project = 'tvm-book' # 项目名称
author = 'xinetzone' # 文档的作者
copyright = '2022, xinetzone' # 版权信息

# == 国际化输出 =======================================================================================
language = 'zh_CN'
locale_dirs = ['../locales/'] # 翻译文件的路径
gettext_compact = False # 为每个翻译创建单独的 .po 文件。

# 通用配置
# =====================================================================================================
# 表示 Sphinx 扩展的模块名称的字符串列表。它们可以是
# Sphinx 自带的插件（命名为 'sphinx.ext.*'）或您自定义的插件。
# -------------------------------------------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx_design",
    'autoapi.extension', # 自动生成API文档
    "sphinx.ext.viewcode", # 添加到高亮源代码的链接
    "sphinx.ext.extlinks", # 缩短外部链接
    "sphinx.ext.intersphinx", # 链接到其他文档
    'sphinx_copybutton', # 为代码块添加复制按钮。
    "sphinx_comments", # 为 Sphinx 文档添加评论和注释功能。
    "sphinx.ext.napoleon", # 支持 Google 和 Numpy 风格的文档字符串
]

# 在此添加包含模板的任何路径，相对于此目录。
# -------------------------------------------------------------------------------------
templates_path = ['_templates']
# 相对于源目录的模式列表，用于匹配在查找源文件时要忽略的文件和目录。
# 此模式还会影响 html_static_path 和 html_extra_path。
# -------------------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
# 链接到其他项目的文档
# -------------------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3.12", None),
    "sphinx": ("https://daobook.github.io/sphinx/", None),
    "pst": ("https://daobook.github.io/pydata-sphinx-theme/", None),
}
# 缩短外部链接
# -------------------------------------------------------------------------------------
extlinks = {
    'daobook': ('https://daobook.github.io/%s', 'Daobook %s'),
    'xinetzone': ('https://xinetzone.github.io/%s', 'xinetzone %s'),
}
# == 配置复制按钮 ====================================================================================
# 使用 ``:not()`` 排除复制按钮出现在笔记本单元格编号上
# 默认的 copybutton 选择器是 `div.highlight pre`
copybutton_exclude = '.linenos, .gp' # 跳过 Pygments 生成的所有提示符
copybutton_selector = ":not(.prompt) > div.highlight pre"

# == HTML 输出 =======================================================================================
# 用于 HTML 和 HTML Help 页面的主题
# -------------------------------------------------------------------------------------
html_theme = 'xyzstyle' # 使用的主题名称
html_logo = "_static/images/logo.jpg"
html_title = "Sphinx xyzstyle Theme"
html_copy_source = True
html_favicon = "_static/images/favicon.jpg"
html_last_updated_fmt = '%Y-%m-%d, %H:%M:%S' # 文档的最后更新时间格式
# 在此添加包含自定义静态文件（如样式表）的任何路径，相对于此目录。
# 它们会在内置静态文件之后被复制，因此名为 "default.css" 的文件将覆盖内置的 "default.css"。
html_static_path = ['_static']
html_css_files = ["css/custom.css", "css/tippy.css"]

# == 主题选项 ========================================================================================
# 选项字典，影响所选主题的外观和感觉。这些选项是特定于主题的。
# -------------------------------------------------------------------------------------
html_theme_options = {
    "use_sidenotes": True,  # 启用侧边注释/页边注释。
    "repository_url": f"https://github.com/xinetzone/{project}",
    "use_repository_button": True,  # 显示“在 GitHub 上查看”按钮。
    "announcement": "👋欢迎进入编程视界！👋", # 公告横幅
    "use_source_button": True,  # 显示“查看源代码”按钮。
    "use_edit_page_button": True,  # 显示“编辑此页”按钮。
    "use_issues_button": True,  # 显示“报告问题”按钮。
    # 图标链接是一组图像和图标，每个图标都链接到一个页面或外部网站。
    # 如果你希望展示社交媒体图标、GitHub 徽章或项目标志，它们会很有帮助。
    "icon_links": icon_links,
}

# 为您的Sphinx网站添加评论和注释功能
# -------------------------------------------------------------------------------------
comments_config = {
   "hypothesis": True,
    # "dokieli": True,
   "utterances": {
      "repo": f"xinetzone/{project}",
      "optional": "config",
   }
}

# 展示丰富的悬停提示
# -------------------------------------------------------------------------------------
extensions.append("sphinx_tippy")
# tippy_enable_mathjax = True
# tippy_anchor_parent_selector = "div.content"
# tippy_logo_svg = Path("tippy-logo.svg").read_text("utf8")
# tippy_custom_tips = {
#     "https://example.com": "<p>This is a custom tip for <a>example.com</a></p>",
#     "https://atomiks.github.io/tippyjs": (
#         f"{tippy_logo_svg}<p>Using Tippy.js, the complete tooltip, popover, dropdown, "
#         "and menu solution for the web, powered by Popper.</p>"
#     ),
# }
tippy_rtd_urls = [
    "https://docs.readthedocs.io/en/stable/",
    "https://www.sphinx-doc.org/zh-cn/master/",
]

# ===================== 可选 ==========================================================
# 用户可以使用 BibTeX 格式的参考文献数据库，并在文档中插入引用和生成参考文献列表。
# -------------------------------------------------------------------------------------
extensions.append('sphinxcontrib.bibtex')
bibtex_bibfiles = ['refs.bib']
# 自动生成 API 文档的路径
# -------------------------------------------------------------------------------------
extensions.append("autoapi.extension")
autoapi_dirs = [f"../src/{project.replace('-', "_")}"]
# 在文档中嵌入 Graphviz 图
# -------------------------------------------------------------------------------------
extensions.append("sphinx.ext.graphviz")
graphviz_output_format = "svg"
inheritance_graph_attrs = dict(
    rankdir="LR",
    fontsize=14,
    ratio="compress",
)
# 在 Sphinx 文档页面中渲染指定 GitHub 仓库的贡献者列表
# -------------------------------------------------------------------------------------
extensions.append('sphinx_contributors')
# 配置用于交互的启动按钮
# -------------------------------------------------------------------------------------
# 这些按钮将在页面底部显示，可用于启动笔记本或演示。
extensions.append("sphinx_thebe")
html_theme_options.update({
    "repository_branch": "main",
    "path_to_docs": "doc",
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    },
})
thebe_config = {
    "repository_url": f"https://github.com/xinetzone/{project}",
    "repository_branch": "main",
    "selector": "div.highlight",
    # "selector": ".thebe",
    # "selector_input": "",
    # "selector_output": "",
    # "codemirror-theme": "blackboard",  # Doesn't currently work
    # "always_load": True,  # To load thebe on every page
}
# 为 Sphinx 文档添加 Open Graph 元数据。
# -------------------------------------------------------------------------------------
extensions.append("sphinxext.opengraph")
ogp_site_url = f"https://{project}.readthedocs.io/zh-cn/latest/"
ogp_social_cards = {
    "site_url": f"{project}lib.readthedocs.io",  # 请将此替换为您的站点 URL
    "image": "_static/images/logo.jpg", # 请确保您的图片是 PNG 或 JPEG 图片，而不是 SVG
    "font": "Noto Sans CJK JP", # 支持中文字体 
    "line_colors": "#4078c0",
}
# 用于生成多版本和多语言的 sitemaps.org 兼容的站点地图
# -------------------------------------------------------------------------------------
extensions.append("sphinx_sitemap")
sitemap_url_scheme = "{lang}{version}{link}"
if not os.environ.get("READTHEDOCS"):
    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "http://127.0.0.1:8000/")
    sitemap_url_scheme = "{link}"
elif os.environ.get("GITHUB_ACTIONS"):
    html_baseurl = os.environ.get("SITEMAP_URL_BASE", "https://xinetzone.github.io/")
sitemap_locales = [None] # 语言列表

# 其他配置
# -------------------------------------------------------------------------------------
nitpick_ignore = [
    ("py:class", "docutils.nodes.document"),
    ("py:class", "docutils.parsers.rst.directives.body.Sidebar"),
]
# application/vnd.plotly.v1+json and application/vnd.bokehjs_load.v0+json
# unknown_mime_type - application/vnd.plotly.v1+json and application/vnd.bokehjs_load.v0+json
# domains - sphinx_proof.domain::prf needs to have `resolve_any_xref` method
# mime_priority - latex priority not set in myst_nb for text/html, application/javascript
suppress_warnings = [
    "mystnb.unknown_mime_type", "mystnb.mime_priority",  # 禁用 application/vnd.plotly.v1+json and application/vnd.bokehjs_load.v0+json 警告
    "myst.xref_missing", "myst.domains", # 禁用 myst 警告
    "ref.ref",
    "autoapi.python_import_resolution", "autoapi.not_readable" # 禁用 autoapi 警告
]
numfig = True
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    # "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]

html_sidebars = {
    "reference/blog/*": [
        "navbar-logo.html",
        "search-field.html",
        "ablog/postcard.html",
        "ablog/recentposts.html",
        "ablog/tagcloud.html",
        "ablog/categories.html",
        "ablog/archives.html",
        "sbt-sidebar-nav.html",
    ]
}

# 排除笔记本不执行
nb_execution_excludepatterns = [
    "read/**/**/*.ipynb",
    "tutorials/**/**/*.ipynb",
    "frontend/**/**/*.ipynb",
    "app/**/**/*.ipynb",
    "topics/**/**/*.ipynb",
]
nb_execution_mode = "off" #"cache"

extensions.append("sphinxcontrib.icon")
