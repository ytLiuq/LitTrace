# Paper PDF Downloader — 学术论文通用下载技能

## 概述

一套方案，覆盖七家主流学术出版商的论文 PDF 下载：**Wiley、Springer Nature、MDPI、IEEE、ACS、Elsevier、RSC**。

核心设计：**输入 DOI 即可自动完成下载，无需手动指定出版商。**

脚本根据 DOI 前缀自动识别出版商，执行三步递进流程：

1. **Unpaywall 查 OA** → 有镜像则 curl 直下（最快、无需浏览器）
2. **stealth 浏览器方案** → 注入 8 项反检测覆盖，通过 Cloudflare 后用 fetch+blob 下载
3. **用户辅助** → Cloudflare Captcha 或 CARSI 登录无法自动通过时，提示用户在浏览器中手动完成

## ⚠️ 执行环境约束（必读）

**此技能不能在沙箱/CI 中独立运行。** 以下三个前置条件，沙箱和 CI 环境均不满足：

| 前置条件 | 沙箱/CI 是否满足 | 原因 |
|---------|-----------------|------|
| CDP 浏览器 | ❌ 不满足 | 沙箱是 Linux 无 GUI，无法启动 `chrome --remote-debugging-port` |
| 网络可达出版社 | ❌ 大概率不满足 | 沙箱可 curl 公网，但 Cloudflare 会识别沙箱 IP 和 TLS 指纹直接拦截 |
| Cloudflare/CARSI 人工验证 | ❌ 不满足 | 需要用户能看到浏览器窗口并手动操作，沙箱无 GUI 无法交互 |

**此技能必须在用户真实 Mac 上执行**，通过 `dumate-browser-use` skill 的 Extension 模式（复用用户已打开的 Chrome 及其登录态）或 CDP 有头模式（在用户 Mac 上启动独立 Chrome）。

Agent 正确执行路径：
1. 通过 `dumate-browser-use` skill 的 `init-extension.sh` 在用户 Mac 上启动 CDP 浏览器
2. 将脚本复制到用户 Mac 本地执行（或在沙箱中通过 CDP WebSocket 远程控制用户 Mac 上的浏览器）
3. 当需要人工验证时，通过对话提示用户在其浏览器窗口中完成操作

> 历史测试全部在用户 Mac（macOS）上通过 `dumate-browser-use` Extension 模式完成，沙箱中从未跑过真实 E2E。

## 触发场景

当用户提到以下内容时使用本技能：

- 下载论文 / 下载 PDF / 下载学术论文
- 获取全文 / 获取 PDF
- 论文下载
- 某个 DOI 对应的论文
- 指定出版商的论文下载（Wiley、Springer Nature、MDPI、IEEE、ACS、Elsevier、RSC）
- 批量下载论文 / 帮我下几篇 XX 领域的论文

关键触发词：下载论文、下载 PDF、论文 PDF、获取全文、DOI 下载、学术论文下载、paper download、fetch PDF、获取文献

使用示例：
- "帮我下载这篇论文 10.1039/d5nr04405g"
- "下载 ACS 上的一篇柔性传感器论文"
- "这个 DOI 的论文能下吗：10.1002/adfm.202414678"
- "帮我下 7 篇不同出版商的压阻传感器论文"

## 前置条件

### 1. CDP 浏览器（必需，仅限用户 Mac）

需启动带远程调试端口的 Chrome 浏览器。脚本通过 CDP（Chrome DevTools Protocol）控制浏览器。

> **沙箱无法满足此条件。** 沙箱是 Linux 无 GUI，无法启动 Chrome。以下命令须在用户 Mac 上执行。

**启动方式 A — 用户 Mac 上直接启动：**

```bash
# macOS
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=19222

# Linux
google-chrome --remote-debugging-port=19222
```

**启动方式 B — 通过 dumate-browser-use skill：**

```bash
source "${_SKILL_DIR}/scripts/init-extension.sh"
# 自动降级为 CDP 有头模式，浏览器地址 http://127.0.0.1:19222
```

**验证 CDP 可用：**

```bash
curl -s http://127.0.0.1:19222/json/version | python3 -c "import sys,json; print(json.load(sys.stdin)['Browser'])"
# 应输出如 "Chrome/144.0.7559.60"
```

### 2. Python 依赖（必需）

```bash
pip install requests websocket-client
```

验证：

```bash
python3 -c "import requests, websocket; print('OK')"
```

### 3. Unpaywall 邮箱（必需）

必须使用真实域名的邮箱。`test@example.com` 会被拒绝（HTTP 422）。

默认使用 `research@sjtu.edu.cn`（上海交通大学）。如需更换：

```bash
python3 scripts/universal_paper_downloader.py <DOI> --email your_email@your_domain.edu
```

### 4. 机构认证（按需）

部分出版商的非 OA 论文需要 CARSI/Shibboleth 机构认证：

- **ACS**：非 OA 论文需 CARSI 登录
- **IEEE**：非 OA 论文需 CARSI 登录或机构 IP
- 脚本会自动检测并提示用户在浏览器窗口中完成登录

CARSI 登录态不持久化——关闭浏览器后失效，下次使用需重新登录。

## 脚本位置

```
{SKILL_DIR}/scripts/universal_paper_downloader.py
```

其中 `{SKILL_DIR}` 为本 SKILL.md 所在目录。

## 用法

```bash
# 基本用法（输出到 ~/Desktop/<DOI>.pdf）
python3 scripts/universal_paper_downloader.py 10.1039/d5nr04405g

# 指定输出路径
python3 scripts/universal_paper_downloader.py 10.1039/d5nr04405g -o ~/Desktop/paper.pdf

# 指定 CDP 地址和邮箱
python3 scripts/universal_paper_downloader.py 10.1021/acsami.5c16055 \
    --cdp http://127.0.0.1:19222 --email research@sjtu.edu.cn

# 完整参数
python3 scripts/universal_paper_downloader.py <DOI> \
    -o <输出路径> \
    --cdp http://127.0.0.1:19222 \
    --email research@sjtu.edu.cn
```

**参数说明：**

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `doi` | 是 | — | 论文 DOI（如 `10.1039/d5nr04405g`） |
| `-o, --output` | 否 | `~/Desktop/<DOI>.pdf` | 输出文件路径 |
| `--cdp` | 否 | `http://127.0.0.1:19222` | CDP 浏览器地址 |
| `--email` | 否 | `research@sjtu.edu.cn` | Unpaywall 查询邮箱 |

## 统一决策流程

```
输入: DOI
  │
  ├─ Step 1: 从 DOI 前缀自动识别出版商
  │   10.1002 → Wiley
  │   10.1038 → Springer Nature
  │   10.3390 → MDPI
  │   10.1109 → IEEE
  │   10.1021 → ACS
  │   10.1016 → Elsevier
  │   10.1039 → RSC
  │
  ├─ Step 2: 查询 Unpaywall 获取 OA 状态
  │   ├─ 有 repository 镜像 → curl 直下（优先，无需浏览器）
  │   │   ├─ 成功 → 完成 ✅
  │   │   └─ 失败 → 进入 Step 3
  │   ├─ 有 publisher OA 链接 → curl 直下试试
  │   │   ├─ 成功 → 完成 ✅（Springer Nature 等无 Cloudflare 的）
  │   │   └─ 失败 → 进入 Step 3
  │   └─ 无 OA → 进入 Step 3
  │
  ├─ Step 3: stealth 浏览器方案
  │   ├─ 创建新标签页，注入 stealth 脚本（8 项覆盖）
  │   ├─ 设置下载目录（Page.setDownloadBehavior）
  │   ├─ 导航到 landing page（doi.org 自动重定向到出版商页面）
  │   │   ├─ Cloudflare managed challenge → stealth 自动通过（最多等 60s）
  │   │   ├─ Cloudflare Captcha → 提示用户手动验证
  │   │   │   → 用户完成后最多再等 120s
  │   │   └─ 无 Cloudflare → 继续
  │   ├─ 检测是否需要 CARSI 登录 → 提示用户完成
  │   │
  │   └─ 下载 PDF（按出版商自动选择最优方法）:
  │       │
  │       ├─ 方法 A: CDP setDownloadBehavior + goto
  │       │   适用: Wiley
  │       │   原理: 设置下载路径后导航到 PDF URL，浏览器自动触发下载
  │       │   流程: setDownloadBehavior → navigate(pdfdirect URL) → 检查文件
  │       │
  │       ├─ 方法 B: fetch + blob + base64 回传
  │       │   适用: MDPI、IEEE、ACS、RSC、Elsevier
  │       │   原理: 在页面内用 JS fetch PDF blob → 转 base64 → 回传 Python 写盘
  │       │   流程: 在 JS 中 fetch(url) → blob → arrayBuffer → btoa(binary) → 返回 Python
  │       │   注意: 同域用相对路径避免 CORS（脚本自动处理）
  │       │   注意: RSC 需先导航到 PDF URL 获取 silverchair CDN 重定向地址
  │       │   注意: Elsevier 需先导航到 pdfft URL，等待重定向到
  │       │         pdf.sciencedirectassets.com CDN 后再 fetch（同域）
  │       │   注意: IEEE 需从页面 HTML 提取 arnumber 构造 stampPDF URL
  │       │
  │       └─ 方法 C: <a download> 触发浏览器原生下载
  │           适用: fetch 失败时的 fallback
  │           原理: 创建 <a> 元素，设置 download 属性，触发 click 事件
  │
  └─ 输出: PDF 文件 + 使用的下载方法
```

## 七家出版商详细策略

### 1. Wiley（DOI 前缀 10.1002）

**Landing page**: `https://advanced.onlinelibrary.wiley.com/doi/{DOI}`

**PDF URL**: `https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/{DOI}?download=true`

**下载方法**: 方法 A — CDP setDownloadBehavior + goto

**流程**:
1. 注入 stealth → 导航到 landing page
2. Wiley 无 Cloudflare 拦截，页面直接加载
3. 设置 `Page.setDownloadBehavior` 指定下载目录
4. 导航到 pdfdirect URL → 浏览器自动触发下载
5. 检查下载目录中的新 PDF 文件

**注意事项**:
- setDownloadBehavior 的 downloadPath 不要设为脚本所在目录（会覆盖脚本自身）
- 下载的文件名由浏览器自动生成，脚本会在下载目录中查找最近 1 分钟内创建的 PDF

### 2. Springer Nature（DOI 前缀 10.1038）

**Landing page**: `https://doi.org/{DOI}`（重定向到 nature.com）

**PDF URL**: `https://www.nature.com/articles/{article_id}.pdf`

**下载方法**: Unpaywall OA → curl 直下（优先）；若无 OA 则 stealth + fetch+blob

**流程**:
1. 查 Unpaywall → 如果是 OA（gold/green），获取 publisher PDF URL
2. curl 直下 — Springer Nature 无 Cloudflare，curl 通常直接成功
3. 如果 curl 失败 → stealth 浏览器 + fetch+blob

**注意事项**:
- Springer Nature 的 `nature.com` 没有 Cloudflare 保护，curl 直下成功率高
- Nature Communications、Scientific Reports 等期刊大多是 OA

### 3. MDPI（DOI 前缀 10.3390）

**Landing page**: `https://doi.org/{DOI}`（重定向到 mdpi.com）

**PDF URL**: 从页面提取，格式为 `https://www.mdpi.com/{ISSN}/{vol}/{issue}/{article}/pdf`

**下载方法**: 方法 B — stealth + fetch+blob（同域相对路径）

**流程**:
1. 注入 stealth → 导航到 landing page
2. MDPI 的 Akamai WAF 阻止 curl 和跨域 fetch
3. 但同域相对路径的 fetch 可以成功
4. 脚本自动将完整 URL 转换为相对路径再 fetch

**注意事项**:
- MDPI 有 Akamai WAF（不是 Cloudflare），stealth 对 Akamai 也有一定效果
- 不要用 setDownloadBehavior（会覆盖脚本文件）
- 必须用相对路径 fetch，跨域 fetch 会被 Akamai 拦截

### 4. IEEE（DOI 前缀 10.1109）

**Landing page**: `https://doi.org/{DOI}`（重定向到 ieeexplore.ieee.org）

**PDF URL**: `https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}`

**下载方法**: Unpaywall → repository（arXiv/大学仓库）curl 直下（优先）；若无 OA 则 stealth + fetch+blob

**流程**:
1. 查 Unpaywall → 如果有 repository 镜像（如 arXiv、大学 DSpace），curl 直下
2. 如果无 OA → stealth 浏览器导航到 IEEE 文章页面
3. 从页面 URL 提取 `arnumber`（格式 `/document/{arnumber}`）
4. 构造 stampPDF URL
5. fetch+blob 下载

**注意事项**:
- IEEE 的 PDF 按钮是 `javascript:void()`，不能直接点击
- 必须从 URL 或 HTML 中提取 `arnumber` 来构造 stampPDF URL
- IEEE OA 论文通常有 arXiv 副本，优先通过 Unpaywall 查找
- 非 OA 论文需要 CARSI 机构认证

### 5. ACS（DOI 前缀 10.1021）

**Landing page**: `https://pubs.acs.org/doi/{DOI}`

**PDF URL**: `https://pubs.acs.org/doi/pdf/{DOI}`

**下载方法**: stealth → Cloudflare 自动通过 → fetch+blob

**流程**:
1. 注入 stealth → 导航到 landing page
2. ACS 使用 Cloudflare managed challenge
3. stealth 8 项覆盖通常能自动通过（约 80% 成功率）
4. 通过后直接 fetch+blob 下载 PDF
5. 如果 Cloudflare 严格时无法自动通过 → 提示用户手动验证

**注意事项**:
- Cloudflare 严格时（约 20% 概率）stealth 无法自动通过，需用户在浏览器窗口中勾选 "I'm not a robot"
- 脚本会先自动等待 60 秒，再提示用户操作，完成后最多再等 120 秒
- OA 论文（ACS Omega 等）免 CARSI 登录
- 非 OA 论文需要 CARSI 机构认证

### 6. Elsevier（DOI 前缀 10.1016）

**Landing page**: `https://doi.org/{DOI}`（重定向到 sciencedirect.com）

**PDF URL**: 从页面提取 `pdfft` 链接，格式为
`https://www.sciencedirect.com/science/article/pii/{PII}/pdfft?md5={hash}&pid=1-s2.0-{PII}-main.pdf`

#### ⚠️ 根因：pdfft → CDN 跳转依赖机构授权，不是等待时间问题

> **这是 agent 最容易误判的地方。** 403 的根因不是"等待时间不够"或"CDN 跳转未完成"，而是**当前浏览器会话没有 ScienceDirect 机构访问权限**。

ScienceDirect 的 pdfft → CDN 重定向是一个**授权检查环节**：

```
导航到 pdfft URL
  │
  ├─ 会话有机构授权（已通过 CARSI/Shibboleth 登录）
  │   │
  │   ├→ 服务端验证 Cookie → 生成 CDN 签名 URL
  │   │   → 重定向到 pdf.sciencedirectassets.com?X-Amz-Signature=...
  │   │   → 返回 PDF ✅
  │   │
  │   └→ 但全新 CDP 会话没有机构 Cookie → 不会发生重定向
  │
  └─ 会话无机构授权（全新 CDP 浏览器，未登录）
      │
      ├→ 服务端检测无授权 → 不生成 CDN 签名 URL
      │   → 返回 403 或 "Get Access" 页面
      │   → 等再久也不会跳转到 CDN ❌
      │
      └→ 页面可能显示:
          "Get Access" / "Purchase PDF" / "Check for this article elsewhere"
```

**历史测试能成功的原因**：Extension 模式复用用户已打开的 Chrome，里面有 SJTU CARSI 登录态的 Cookie。全新 CDP 会话（`init-extension.sh` 降级启动的独立 Chrome）没有这个 Cookie，授权检查直接失败。

#### 正确流程（Agent 必须按此顺序执行）

> ⚠️ 脚本的流程是"先提取 pdfft → 导航 → fetch"，**缺少授权检查**。Agent 需要在导航到 pdfft 之前，先确认会话是否有 ScienceDirect 机构访问权限。

**Step 0: 先查 Unpaywall——如果是 OA 论文，直接 curl 下载，跳过整个浏览器方案**

```python
# Step 0: Unpaywall 查 OA
oa_result = query_unpaywall(doi, email)
if oa_result and oa_result.get("is_oa"):
    # 有 repository 镜像 → curl 直下（无需浏览器、无需授权）
    repo_url = oa_result.get("best_oa_location", {}).get("url_for_pdf")
    if repo_url:
        success, info = try_curl_download(repo_url, output_path)
        if success:
            return True, f"Unpaywall OA → curl ({info})"
    # 有 publisher OA → 也试试 curl
    pub_url = oa_result.get("best_oa_location", {}).get("url")
    if pub_url:
        success, info = try_curl_download(pub_url, output_path)
        if success:
            return True, f"Unpaywall publisher OA → curl ({info})"
```

> Elsevier 非 OA 论文占比很高（约 70%），必须走下面的授权流程。

**Step 1: 导航到 landing page，检查是否有机构访问权限**

```python
# 导航到 ScienceDirect 文章页面
browser.navigate(f"https://doi.org/{doi}", wait=20)

# 检查页面是否显示 "Get Access"（无机构权限）
check_access_js = """
(function() {
    var body = document.body.innerText || '';
    // 无授权的标志
    if (body.includes('Get Access') || body.includes('Purchase PDF')
        || body.includes('Check for this article elsewhere')) {
        return 'no_access';
    }
    // 有授权的标志——页面有 "Download PDF" 按钮或 pdfft 链接
    var pdfLink = document.querySelector('a[href*="pdfft"], a[aria-label*="PDF"], a[title*="PDF"]');
    if (pdfLink) return 'has_access';
    // 也检查页面是否有 "Download" 按钮
    if (body.includes('Download') && body.includes('PDF')) return 'has_access';
    return 'unknown';
})()
"""
result = browser.send("Runtime.evaluate", {"expression": check_access_js})
access_status = result.get("result", {}).get("result", {}).get("value", "unknown")
```

**Step 2a: 如果无机构访问权限 → 引导用户完成 CARSI 登录**

```python
if access_status == "no_access":
    print("  ⚠ ScienceDirect 检测到无机构访问权限")
    print("  需要通过 CARSI/Shibboleth 完成机构登录")

    # 导航到 ScienceDirect 的机构登录入口
    # 方法 1: 点击页面上的 "Get Access" → "Sign in via your institution"
    browser.send("Runtime.evaluate", {"expression": """
        (function() {
            // 查找 "Get Access" 或 "Sign in" 按钮并点击
            var btns = document.querySelectorAll('a, button');
            for (var b of btns) {
                var text = (b.innerText || '').toLowerCase();
                if (text.includes('get access') || text.includes('sign in')
                    || text.includes('institution')) {
                    b.click();
                    return 'clicked: ' + text;
                }
            }
            return 'no_button_found';
        })()
    """})
    time.sleep(5)

    # 方法 2: 直接导航到 ScienceDirect 的 Shibboleth 登录页
    # browser.navigate("https://www.sciencedirect.com/user/login", wait=15)

    print("  请在浏览器中完成 SJTU CARSI 登录:")
    print("    1. 选择 'Sign in via your institution'")
    print("    2. 搜索 'Shanghai Jiao Tong University'")
    print("    3. 完成 SJTU 统一身份认证")
    _wait_for_user("  >>> 完成 CARSI 登录后按 Enter 继续...")

    # 登录完成后，重新导航到文章页面
    browser.navigate(f"https://doi.org/{doi}", wait=20)
    # 再次检查权限
    result = browser.send("Runtime.evaluate", {"expression": check_access_js})
    access_status = result.get("result", {}).get("result", {}).get("value", "unknown")

    if access_status == "no_access":
        print("  ⚠ 登录后仍无访问权限")
        print("  → 尝试 Unpaywall 查找 OA 镜像作为 fallback")
        # fallback 到 Unpaywall 或告知用户无法下载
```

> **CARSI 登录态不持久**：关闭浏览器后 Cookie 失效。每次使用全新 CDP 浏览器都需重新登录。Extension 模式（复用用户 Chrome）通常已有登录态。

**Step 2b: 如果有机构访问权限 → 提取 pdfft → 导航 → 等待 CDN → fetch**

```python
if access_status == "has_access":
    # 从页面提取 pdfft 链接
    # （用脚本中已有的 extract_js，此处不重复）
    pdf_url = extract_pdfft_from_page(browser)

    if not pdf_url:
        return False, "无法提取 pdfft 链接"

    # 导航到 pdfft URL
    browser.send("Page.navigate", {"url": pdf_url})

    # 轮询等待 CDN 重定向——最多 30 秒
    cdn_url = None
    for i in range(15):
        time.sleep(2)
        current = browser.get_url()
        if "sciencedirectassets" in current or "els-cdn" in current:
            cdn_url = current
            time.sleep(3)  # 等待页面完全加载
            break
        # 也检查 contentType
        ct = browser.send("Runtime.evaluate", {"expression": "document.contentType || ''"})
        if ct.get("result", {}).get("result", {}).get("value") == "application/pdf":
            cdn_url = current
            time.sleep(3)
            break

    if cdn_url:
        # 确认当前页面在 CDN 域名上，同域 fetch
        success, info = browser.fetch_blob_and_save(cdn_url, output_path)
        if success:
            return True, f"fetch+blob ({info / 1024 / 1024:.1f}MB)"

    # fetch 失败 → Page.printToPDF fallback
    if cdn_url or access_status == "has_access":
        print("  → fetch 失败，尝试 Page.printToPDF...")
        result = browser.send("Page.printToPDF", {
            "printBackground": True,
            "preferCSSPageSize": True
        })
        pdf_b64 = result.get("result", {}).get("data", "")
        if pdf_b64:
            pdf_bytes = base64.b64decode(pdf_b64)
            with open(output_path, "wb") as f:
                f.write(pdf_bytes)
            return True, f"Page.printToPDF ({len(pdf_bytes) / 1024 / 1024:.1f}MB)"
```

#### 授权检查决策流程

```
Elsevier DOI
  │
  ├─ Step 0: Unpaywall 查 OA
  │   ├─ 有 repository 镜像 → curl 直下 ✅（无需浏览器）
  │   └─ 无 OA → 进入 Step 1
  │
  ├─ Step 1: 导航到 landing page，检查机构访问权限
  │   │
  │   ├─ 有权限（页面有 "Download PDF" / pdfft 链接）
  │   │   └→ Step 2b: 提取 pdfft → 导航 → CDN 重定向 → fetch+blob ✅
  │   │
  │   └─ 无权限（页面显示 "Get Access" / "Purchase PDF"）
  │       └→ Step 2a: 引导用户 CARSI 登录
  │           ├─ 登录后重新检查权限
  │           │   ├─ 有权限 → Step 2b ✅
  │           │   └─ 仍无权限 → Unpaywall fallback 或告知用户
  │           └─ 登录失败 → Unpaywall fallback 或告知用户
  │
  └─ Fallback: 无法获取机构授权时
      ├─ Unpaywall 查 repository 镜像（大学 DSpace 仓库等）
      ├─ Google Scholar 查 preprint 版本
      └─ 告知用户需手动在有权限的浏览器中下载
```

#### 关键注意事项

- **pdfft → CDN 重定向依赖机构授权**——没有 CARSI/Shibboleth 登录态，ScienceDirect 不会生成 CDN 签名 URL，导航到 pdfft 只会返回 403 或 "Get Access" 页面
- **CDN 签名 URL 时效性**——通常 4-24 小时，需在过期前 fetch
- **CARSI 登录态不持久**——关闭浏览器后 Cookie 失效，全新 CDP 会话需重新登录
- **Extension 模式 vs CDP 有头模式**：
  - Extension 模式（复用用户 Chrome）→ 通常已有 CARSI 登录态，直接走 Step 2b
  - CDP 有头模式（`init-extension.sh` 降级启动）→ **通常没有登录态**，需走 Step 2a
- **PII 号从 doi.org 重定向获取**，不需要手动构造
- **`Page.printToPDF` 只在确认有机构权限后使用**——无权限时 printToPDF 导出的是 "Get Access" 页面而非 PDF

#### 与脚本实现的差异

| 步骤 | 脚本实现 | Agent 应做的 |
|------|---------|-------------|
| 授权检查 | ❌ 无——直接提取 pdfft 就导航 | ✅ 先检查页面是否显示 "Get Access"，有则引导 CARSI 登录 |
| Unpaywall 预查 | ✅ 有（但 OA 时不走浏览器） | ✅ 同，但应作为无权限时的 fallback |
| CARSI 登录引导 | ❌ 无——只检测关键词但无引导 | ✅ 检测无权限后，引导用户走 CARSI/Shibboleth 登录 |
| 导航到 pdfft | `navigate(pdf_url, wait=10)` | 确认有权限后再导航，轮询 30 秒等 CDN |
| fetch 403 fallback | 尝试 `<a download>` | `Page.printToPDF`（但仅在有权限时有效） |

### 7. RSC（DOI 前缀 10.1039）

**Landing page**: `https://doi.org/{DOI}`（重定向到 pubs.rsc.org）

**PDF URL**: `https://pubs.rsc.org/en/content/articlepdf/{year}/{journal}/{article_code}`

**下载方法**: stealth → landing page → 提取 PDF URL → 导航到 PDF URL → silverchair CDN → fetch+blob

**流程**:
1. 注入 stealth → 导航到 doi.org（重定向到 RSC 文章页面）
2. RSC 使用 Cloudflare managed challenge，stealth 通常能自动通过
3. 从页面提取 `articlepdf` 链接，或从 DOI 构造 PDF URL
4. **关键步骤**：导航到 PDF URL → 浏览器重定向到 `rscj.silverchair-cdn.com` CDN
5. 从 CDN 同域 fetch+blob 下载

**注意事项**:
- RSC 的 PDF URL 需要年份信息，从 DOI 的文章代码中提取
  - DOI 格式：`10.1039/d5nr04405g`，文章代码 `d5nr04405g`
  - 年份映射：`d5` → 2025, `d4` → 2024, `d3` → 2023...
  - 期刊代码：`nr` → Nanoscale, `ma` → Materials Advances...
- 导航到 PDF URL 后必须检查是否重定向到了 silverchair CDN
- CDN URL 格式：`https://rscj.silverchair-cdn.com/rscj/content_public/journal/{journal}/{vol}/{issue}/10.1039_{article_code}/1/{article_code}.pdf`

## 出版商路径速查表

| 出版商 | DOI 前缀 | 典型路径 | 需要用户交互 |
|--------|---------|---------|-------------|
| Wiley | 10.1002 | stealth → setDownloadBehavior + goto | 否 |
| Springer Nature | 10.1038 | Unpaywall OA → curl 直下 | 否 |
| MDPI | 10.3390 | stealth → fetch+blob（相对路径） | 否 |
| IEEE | 10.1109 | Unpaywall → repository curl 直下；若无 OA → stealth + fetch+blob | 非 OA 需 CARSI |
| ACS | 10.1021 | stealth → Cloudflare 自动通过 → fetch+blob | Cloudflare 严格时需手动验证 |
| Elsevier | 10.1016 | Unpaywall OA → curl 直下；非 OA → 检查机构权限 → CARSI 登录 → pdfft → CDN → fetch+blob | 非 OA 需 CARSI |
| RSC | 10.1039 | stealth → landing → silverchair CDN → fetch+blob | 否 |

> **关键洞察**：
> - **Elsevier 403 的根因是会话无机构授权**，不是等待时间问题。pdfft → CDN 重定向依赖 CARSI/Shibboleth 登录态——全新 CDP 浏览器没有 Cookie，ScienceDirect 不生成 CDN 签名 URL，等再久也不会跳转。历史测试成功是因为 Extension 模式复用了用户已登录的 Chrome。Agent 必须先检查权限，无权限时引导 CARSI 登录
> - ACS 的 Cloudflare managed challenge 可被 stealth 8 项覆盖自动通过，但偶尔严格时需用户手动验证（脚本自动等待 120 秒）
> - IEEE 的 OA 论文通常有 arXiv 副本
> - MDPI 的 Akamai WAF 阻止 curl 和跨域 fetch，但同域相对路径 fetch 可成功
> - RSC 和 Elsevier 都有 CDN 重定向：RSC → silverchair CDN，Elsevier → sciencedirectassets CDN（但 Elsevier 的 CDN 重定向需要机构授权）

## stealth 注入脚本

脚本中内置的 stealth JS 包含 **8 项基础覆盖**（见 `STEALTH_JS` 常量），通过 `Page.addScriptToEvaluateOnNewDocument` 注入。这 8 项只在 **JS 层面**修改属性，在 Extension 模式（复用用户已打开的 Chrome）下通常够用。

但在 CDP 有头模式下，Cloudflare 会在 **HTTP 层面**直接拦截——这是你的 agent 遇到"Cloudflare 无法通过"的根因。

### 根因：JS 层面 vs HTTP 层面

| 层面 | 检测什么 | 脚本 8 项是否覆盖 | 后果 |
|------|---------|-----------------|------|
| JS 层面 | navigator.webdriver、plugins、chrome 等 | ✅ 覆盖 | 不覆盖会被 Cloudflare JS 指纹检测 |
| HTTP 层面 | User-Agent 请求头中的 `HeadlessChrome` | ❌ 未覆盖 | Cloudflare 在 HTTP 层直接识别并拦截，JS 层 stealth 完全无效 |

> **CDP 有头模式启动的 Chrome，UA 中包含 `HeadlessChrome`**。Cloudflare 从 HTTP 请求头（而非 JS）就能识别。JS 层面修改 `navigator.userAgent` **不影响** HTTP 请求头——必须通过 CDP 的 `Network.setUserAgentOverride` 在协议层面修正。

### Agent 必须做的增强（脚本未内置）

> ⚠️ 以下内容脚本**没有实现**，是 agent 在执行时需要自行补充的。脚本只做了 8 项 JS 覆盖和 `addScriptToEvaluateOnNewDocument` 注入，**没有调用** `Page.enable`/`Network.enable`/`Network.setUserAgentOverride`。

#### 增强 1：启用 Page/Network 域 + UA 修正

在 `connect_new_tab()` 之后、`inject_stealth()` 之前，agent 需要执行：

```python
# 1. 启用 Page 和 Network 域
browser.send("Page.enable")
browser.send("Network.enable")

# 2. 检测并修正 UA（HTTP 层面）
ua_result = browser.send("Runtime.evaluate", {"expression": "navigator.userAgent"})
current_ua = ua_result.get("result", {}).get("result", {}).get("value", "")
if current_ua and "HeadlessChrome" in current_ua:
    fixed_ua = current_ua.replace("HeadlessChrome", "Chrome")
    browser.send("Network.setUserAgentOverride", {"userAgent": fixed_ua})
    print(f"  [stealth] UA 已修正: HeadlessChrome → Chrome")
```

> **为什么脚本没有内置**：这些增强是在历史测试发现问题后总结的，脚本源码尚未更新。agent 应在执行时自行加入。

#### 增强 2：将 8 项 JS 覆盖扩展为 13 项

脚本内置的 8 项覆盖在 CDP 有头模式下不够。以下是**应补充的 5 项**（追加到 `STEALTH_JS` 末尾，`})();` 之前）：

```javascript
// 9. navigator.hardwareConcurrency
// HeadlessChrome 默认为 1，正常浏览器通常 ≥ 4
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 8 });

// 10. navigator.deviceMemory
// HeadlessChrome 可能没有此属性
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8 });

// 11. navigator.platform
// macOS 用户应为 MacIntel
Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });

// 12. navigator.userAgentData（Chrome 90+ 的 UA-CH）
// HeadlessChrome 可能缺少此属性
if (navigator.userAgentData === undefined) {
    Object.defineProperty(navigator, 'userAgentData', {
        get: () => ({
            brands: [
                { brand: 'Google Chrome', version: '144' },
                { brand: 'Chromium', version: '144' },
                { brand: 'Not?A_Brand', version: '99' }
            ],
            mobile: false,
            platform: 'macOS'
        })
    });
}

// 13. UA 字符串中的 HeadlessChrome → Chrome（JS 层面，HTTP 层面需 setUserAgentOverride）
const originalUA = navigator.userAgent;
if (originalUA && originalUA.includes('HeadlessChrome')) {
    Object.defineProperty(navigator, 'userAgent', {
        get: () => originalUA.replace('HeadlessChrome', 'Chrome')
    });
}
```

> **注意**：第 13 项只是 JS 层面的 UA 修正，**不能替代** `Network.setUserAgentOverride`。两者必须同时做。

#### 增强 3：window.chrome 伪造更完整

脚本内置的第 2 项是 `window.chrome = { runtime: {} }`，但 Cloudflare 会检查 chrome 对象的多层属性。应替换为：

```javascript
window.chrome = {
    runtime: {
        OnInstalledReason: { CHROME_UPDATE: 'chrome_update', INSTALL: 'install', UPDATE: 'update' },
        PlatformArch: { ARM: 'arm', X86_32: 'x86-32', X86_64: 'x86-64' },
        PlatformOs: { ANDROID: 'android', CROS: 'cros', LINUX: 'linux', MAC: 'mac', WIN: 'win' }
    },
    loadTimes: function() { return {}; },
    csi: function() { return {}; }
};
```

#### 增强 4：navigator.plugins 伪造更真实

脚本内置的第 4 项是 `Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] })`，但 Cloudflare 可能检查 Plugin 对象的 `name`/`description`/`filename` 属性。应替换为：

```javascript
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const makePlugin = (name, filename, description) => {
            const p = { name, filename, description, length: 1 };
            p[0] = { type: 'application/pdf', suffixes: 'pdf', description: '' };
            return p;
        };
        const arr = [
            makePlugin('PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
            makePlugin('Chrome PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
            makePlugin('Chromium PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
            makePlugin('Microsoft Edge PDF Viewer', 'internal-pdf-viewer', 'Portable Document Format'),
            makePlugin('WebKit built-in PDF', 'internal-pdf-viewer', 'Portable Document Format')
        ];
        arr.namedItem = function(name) { return this.find(p => p.name === name) || null; };
        arr.refresh = function() {};
        return arr;
    }
});
```

### 完整的增强后 stealth 流程

```
connect_new_tab()
  │
  ├─ Page.enable              ← agent 补充
  ├─ Network.enable           ← agent 补充
  ├─ 检测 UA → Network.setUserAgentOverride  ← agent 补充
  │
  ├─ inject_stealth()
  │   └─ Page.addScriptToEvaluateOnNewDocument
  │       ├─ 8 项基础覆盖（脚本内置）
  │       └─ 5 项增强覆盖（agent 补充到 STEALTH_JS）
  │
  ├─ navigate(landing_url)
  │   └─ Cloudflare managed challenge
  │       ├─ stealth 自动通过（CDP 有头模式需上述全部增强）
  │       └─ 无法自动通过 → 提示用户手动验证
  │
  └─ fetch+blob 下载 PDF
```

### 适用性

| 浏览器启动方式 | UA 是否含 HeadlessChrome | 8 项是否够用 | 需要的增强 |
|-------------|------------------------|-------------|-----------|
| Extension 模式（复用用户 Chrome） | 否 | 通常够用 | 无 |
| CDP 有头模式（`init-extension.sh` 降级） | 是 | **不够** | 增强 1+2+3+4 全部需要 |

### Cloudflare 无法通过时的诊断步骤

当 stealth 注入后 Cloudflare 仍然无法通过时，按以下顺序排查：

1. **检查 UA 是否被修正**（最常见原因）：
   ```python
   result = browser.send("Runtime.evaluate", {"expression": "navigator.userAgent"})
   ua = result.get("result", {}).get("result", {}).get("value", "")
   # 如果 ua 仍包含 "HeadlessChrome" → Network.setUserAgentOverride 未执行或未生效
   ```
   > 这是 agent 遇到"Cloudflare 无法通过"的最常见原因。脚本未内置 UA 修正，agent 必须自行补充。

2. **确认浏览器启动方式**：
   - Extension 模式（复用用户 Chrome）→ UA 通常已是正常 Chrome，stealth 成功率最高
   - CDP 有头模式（`init-extension.sh` 降级启动）→ UA 可能含 HeadlessChrome，**必须**做 UA 修正
   - 建议在启动 Chrome 时直接加 `--user-agent="Mozilla/5.0 ... Chrome/144 ..."` 从源头修正

3. **检查 Cloudflare 验证类型**：
   - managed challenge（自动 JS 验证）→ stealth + UA 修正可自动通过
   - interactive challenge（需勾选 "I'm not a robot"）→ stealth 无法自动通过，需用户手动
   - 脚本会先等 60s 自动通过，再提示用户操作

4. **检查 TLS 指纹**：
   - Cloudflare 可能通过 JA3/JA4 TLS 指纹识别自动化工具
   - CDP 有头模式的 Chrome 的 TLS 指纹与正常 Chrome 相同（都是 Chromium TLS 栈），通常不会被拦
   - 但如果沙箱通过 curl/Python requests 访问，TLS 指纹完全不同，会被直接拦截

## CDPBrowser API 参考

脚本封装了 `CDPBrowser` 类，通过 WebSocket 连接 CDP 浏览器。以下是核心方法：

### 连接管理

| 方法 | 用途 | 返回值 |
|------|------|--------|
| `CDPBrowser(cdp_url)` | 构造函数，指定 CDP 地址 | CDPBrowser 实例 |
| `connect_new_tab()` | 创建新标签页并连接 WebSocket | tab dict（含 id, url, webSocketDebuggerUrl） |
| `close()` | 关闭 WebSocket 连接 | — |

> **注意**：`CDPBrowser` 不提供 `connect_existing_tab` 方法。如需连接已有标签页，可用 `send("Target.getTargets")` 列出后通过 WebSocket URL 手动连接。

### CDP 命令

| 方法 | 用途 | 参数 | 返回值 |
|------|------|------|--------|
| `send(method, params)` | 发送底层 CDP 命令（带 3 次自动重连） | method: CDP 方法名, params: 参数 dict | CDP 响应 dict |

**自动重连机制**：导航后 WebSocket 可能断连。`send` 方法内置 3 次重连逻辑：
1. 捕获异常 → 等待 2 秒
2. 查找原标签页 → 重新连接 WebSocket
3. 如果原标签页不存在 → 创建新标签页 → 重新注入 stealth 和下载路径
4. 重试发送命令

### 页面操作

| 方法 | 用途 | 参数 | 返回值 |
|------|------|------|--------|
| `inject_stealth()` | 注入 8 项 stealth 覆盖到所有未来文档 | — | — |
| `set_download_path(path)` | 设置浏览器下载目录 | path: 目录路径 | — |
| `navigate(url, wait)` | 导航到 URL 并等待 | url: URL, wait: 等待秒数（默认 15） | — |
| `get_url()` | 获取当前页面 URL | — | URL 字符串 |
| `get_title()` | 获取页面标题 | — | 标题字符串 |
| `get_body_text(max_chars)` | 获取页面正文前 N 字符 | max_chars: 最大字符数（默认 500） | 正文文本 |

### Cloudflare 检测

| 方法 | 用途 | 返回值 |
|------|------|--------|
| `is_cloudflare_challenge()` | 检测是否在 Cloudflare 验证页 | bool |
| `wait_for_cloudflare(max_wait, check_interval)` | 轮询等待 Cloudflare 通过 | bool（是否成功） |

**Cloudflare 检测指标**：检查页面标题和正文是否包含以下关键词：
- "请稍候"（中文 Cloudflare 页面）
- "Just a moment"（英文 Cloudflare 页面）
- "安全验证"
- "Are you a robot"
- "captcha"

### PDF 下载

| 方法 | 用途 | 参数 | 返回值 |
|------|------|------|--------|
| `fetch_blob_to_file(pdf_url, output_path)` | 同域 fetch PDF + base64 回传写盘 | pdf_url: PDF URL, output_path: 输出路径 | (success: bool, info: str/int) |
| `trigger_anchor_download(pdf_url, filename)` | `<a download>` 触发浏览器原生下载 | pdf_url: PDF URL, filename: 文件名 | — |

**fetch_blob_to_file 内部流程**：
1. 获取当前页面 URL
2. 如果 PDF URL 与当前页面同域，转换为相对路径（避免 CORS）
3. 在页面中执行 JS：`fetch(url) → blob → arrayBuffer → btoa(binary)` → 返回 base64
4. Python 端 base64 解码 → 写入文件
5. 返回 (True, 文件大小) 或 (False, 错误信息)

## 工作流步骤（Agent 执行指南）

### 1. 确认输入

用户提供 DOI（如 `10.1039/d5nr04405g`）。

如果用户只给了论文标题，先用 CrossRef API 查 DOI：

```bash
curl -s "https://api.crossref.org/works?query.bibliographic=<标题>&rows=1&select=DOI,title" | python3 -c "
import sys, json; d=json.load(sys.stdin); print(d['message']['items'][0]['DOI'])"
```

如果用户要求下载某个主题的论文但没给 DOI，可以用 CrossRef 批量搜索：

```bash
curl -s "https://api.crossref.org/works?query=<关键词>&rows=100&select=DOI,title" | python3 -c "
import sys, json
items = json.load(sys.stdin)['message']['items']
for item in items[:10]:
    print(f\"{item['DOI']}: {item.get('title',[''])[0][:80]}\")"
```

### 2. 检查前置条件

```bash
# 检查 CDP 浏览器
curl -s http://127.0.0.1:19222/json/version | python3 -c "import sys,json; print('OK')"

# 检查 Python 依赖
python3 -c "import requests, websocket; print('OK')"
```

如果 CDP 浏览器未运行：
- 方式 A：提示用户在 Mac 终端运行 `/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=19222`
- 方式 B：通过 `dumate-browser-use` skill 的 `init-extension.sh` 自动启动 CDP 有头模式

### 3. 执行脚本

```bash
python3 {SKILL_DIR}/scripts/universal_paper_downloader.py <DOI> \
    -o <输出路径> \
    --cdp http://127.0.0.1:19222 \
    --email research@sjtu.edu.cn
```

### 4. 处理用户交互

脚本在以下情况会暂停等待用户操作：

**Cloudflare Captcha**（ACS/Elsevier）：
- 脚本先自动等待 60 秒让 stealth 尝试通过
- 如果未通过，输出提示：`请在弹出的浏览器窗口中手动完成验证`
- 交互模式：等待用户按 Enter
- 非交互模式：自动等待 60 秒
- 用户操作后最多再等 120 秒确认通过

**CARSI 登录**（ACS/IEEE 非 OA 论文）：
- 脚本检测到页面需要机构认证时输出提示：`请在浏览器中完成 SJTU CARSI 登录`
- 交互模式：等待用户按 Enter
- 非交互模式：自动等待 60 秒

### 5. 验证结果

```bash
# 检查文件是否是真实 PDF
file <输出路径>
# 应输出 "PDF document, version X.X"

# 检查文件大小
ls -lh <输出路径>
# 正常论文 PDF 通常在 0.5MB ~ 15MB 之间
```

### 6. 批量下载

如果用户要求下载多篇论文：

```bash
# 循环下载
for doi in "10.1002/mame.202400237" "10.1038/srep14751" "10.3390/s23052443"; do
    python3 {SKILL_DIR}/scripts/universal_paper_downloader.py "$doi" \
        -o ~/Desktop/papers/$(echo $doi | tr '/' '_').pdf \
        --cdp http://127.0.0.1:19222 --email research@sjtu.edu.cn
done
```

批量下载时建议：
- 每篇之间间隔 5-10 秒，避免触发反爬
- ACS 论文如果遇到 Cloudflare 严格模式，可能需要逐篇手动验证
- 输出到不同文件名避免覆盖

## 故障排查

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| `CDP 浏览器未运行` | Chrome 未启动或端口不对 | 启动 Chrome `--remote-debugging-port=19222`，用 `--cdp` 指定正确端口 |
| `Unpaywall 拒绝请求 (HTTP 422)` | 邮箱无效 | 用真实域名邮箱 `--email research@sjtu.edu.cn` |
| `Unpaywall 返回 HTTP 404` | DOI 未被索引 | 论文太新或 DOI 有误，进入浏览器方案 |
| `下载到的是 HTML 而非 PDF` | Cloudflare/Akamai 拦截 curl | 正常，脚本会自动进入浏览器方案 |
| `fetch 失败: Failed to fetch` | CORS 跨域 | 脚本自动用相对路径重试；如果仍失败检查是否在同域 |
| `fetch 失败: HTTP 403` | 未通过 Cloudflare 或未登录 | 检查是否需要 CARSI 登录；或手动通过 Cloudflare |
| `WebSocket 断连` | 导航后连接丢失 | 脚本有 3 次自动重连机制，通常自动恢复 |
| `Cloudflare Captcha 无法自动通过` | ACS/Elsevier 的 Captcha | 需用户在浏览器中手动完成验证，脚本自动等待最多 120 秒 |
| `无法找到 PDF 下载链接` | 页面结构变化或需要登录 | 检查是否需要 CARSI 登录；手动在浏览器中找到 PDF 链接 |
| IEEE `javascript:void()` | PDF 按钮是 JS 触发 | 脚本从 URL/HTML 提取 `arnumber` 构造 stampPDF URL |
| MDPI `setDownloadBehavior` 覆盖了脚本文件 | 下载路径与脚本同目录 | 用 `-o /tmp/output.pdf` 指定输出到其他目录 |
| Elsevier `fetch 返回 HTTP 403` | **会话无机构授权**（最常见根因） | pdfft → CDN 重定向依赖 CARSI/Shibboleth 登录态。全新 CDP 浏览器无 Cookie → 不生成 CDN 签名 URL → 403。需先检查权限，引导用户 CARSI 登录 |
| Elsevier `pdfft 返回 HTML` | pdfft 是 JS 跳转页非直接 PDF | 需导航到 pdfft URL，轮询 30 秒等 CDN 重定向（但前提是会话已有机构授权） |
| Elsevier `页面显示 Get Access` | 无机构访问权限 | 需通过 CARSI/Shibboleth 登录，或改用 Unpaywall 查 OA 镜像 |
| Elsevier `未检测到 CDN 重定向` | 无机构授权 或 pdfft URL 错误 | 先检查页面是否显示 "Get Access"；若已登录则检查 pdfft URL 是否正确提取 |
| Elsevier `Page not found` | PII 号不正确 | 脚本通过 doi.org 重定向获取正确的 PII，不要手动构造 |
| RSC `无法提取 PDF URL` | 页面结构变化 | 脚本从 DOI 文章代码构造 PDF URL（年份+期刊+文章代码） |
| 下载的 PDF 大小为 0 或过小 | 下载不完整 | 检查网络连接；重试；检查 CDN URL 是否已过期 |

## 已知限制

### 执行环境限制（根本性）

1. **不能在沙箱/CI 中独立运行**：CDP 浏览器需要 GUI、出版社网站会拦截沙箱 IP、Cloudflare/CARSI 需要人工交互——三者沙箱均不满足。必须在用户 Mac 上通过 `dumate-browser-use` Extension 模式执行
2. **沙箱中从未跑过真实 E2E**：历史 7/7 测试全部在用户 Mac 上完成。沙箱中仅能执行 Unpaywall 查询 + curl 直下（Step 1），Step 2/3 必须在用户 Mac 上执行
3. **网络环境依赖用户 Mac**：Extension 模式复用用户真实浏览器的网络栈和 Cookie，才能访问出版社网站。沙箱直接 curl 会被 Cloudflare 拦截

### 技术限制

4. **ACS Cloudflare 波动**：stealth 8 项覆盖通常能自动通过 Cloudflare managed challenge，但 Cloudflare 严格时（约 20% 概率）无法自动通过，需用户在浏览器中手动勾选验证。脚本会先自动等待 60 秒，再提示用户操作，完成后最多再等 120 秒
5. **IEEE 非 OA 论文**：需要机构 IP 或 CARSI 认证，脚本会提示用户完成登录
6. **CARSI 登录态不持久**：关闭浏览器后失效，下次使用需重新登录
7. **Unpaywall 覆盖延迟**：最新发表的论文可能尚未被 Unpaywall 索引（通常 1-2 周延迟）
8. **Elsevier PII 获取**：脚本从 doi.org 重定向获取 ScienceDirect 的 PII 号，确保导航到正确的文章页面
9. **CDN URL 时效性**：Elsevier 的 `pdf.sciencedirectassets.com` CDN URL 是 AWS CloudFront 签名的，有时效性（通常几小时），需在过期前完成 fetch
10. **不支持非七家出版商**：目前仅支持 Wiley、Springer Nature、MDPI、IEEE、ACS、Elsevier、RSC。其他出版商（如 Taylor & Francis、SAGE、Oxford University Press 等）不在支持范围内

## 测试验证

本技能已于 2026-07-10 在**用户 Mac（macOS）**上通过 `dumate-browser-use` Extension 模式完成两轮七家出版商全量测试，成功率 7/7。

> **注意**：以下测试全部在用户真实 Mac 上执行，沙箱中从未跑过真实 E2E。沙箱中仅能执行 Step 1（Unpaywall 查询 + curl 直下），Step 2/3 依赖用户 Mac 上的 CDP 浏览器。

### 第二轮测试（柔性薄膜压阻传感器论文，修复后）

| 出版商 | 测试 DOI | 大小 | 下载方法 | 人工介入 |
|--------|---------|------|---------|---------|
| Wiley | 10.1002/mame.202400237 | 5.5MB | stealth + setDownloadBehavior + goto | 否 |
| Springer Nature | 10.1038/srep14751 | 5.9MB | Unpaywall OA → curl 直下 | 否 |
| MDPI | 10.3390/s23052443 | 5.9MB | stealth + fetch+blob | 否 |
| IEEE | 10.1109/SENSORS43011.2019.8956652 | 0.8MB | Unpaywall → repository 镜像 curl 直下 | 否 |
| ACS | 10.1021/acsomega.3c04786 | 5.8MB | stealth + Cloudflare 自动通过 + fetch+blob | 否 |
| Elsevier | 10.1016/j.matdes.2025.114201 | 12.7MB | stealth → pdfft → CDN 重定向 → fetch+blob | 否 |
| RSC | 10.1039/d2ma00987k | 2.0MB | stealth → silverchair CDN → fetch+blob | 否 |

> 第二轮修复内容：
> - **Elsevier**：新增 pdfft → `pdf.sciencedirectassets.com` CDN 重定向逻辑，无需人工介入
> - **ACS**：Cloudflare 等待时间从 30s 延长至 120s，非交互模式从 30s 延长至 60s

### 第一轮测试（初始验证）

| 出版商 | 测试 DOI | 大小 | 下载方法 |
|--------|---------|------|---------|
| Springer Nature | 10.1038/s41467-023-36307-4 | 188K | fetch+blob |
| Elsevier | 10.1016/j.carbon.2025.121139 | 8.9M | repository 镜像 curl 直下 |
| Wiley | 10.1002/adfm.202414678 | 8.9M | setDownloadBehavior + goto |
| MDPI | 10.3390/nano16140842 | 5.5M | fetch+blob（相对路径） |
| IEEE | 10.1109/jproc.2026.3679470 | 13M | arXiv repository curl 直下 |
| ACS | 10.1021/acsnano.6c02465 | 9.5M | stealth + fetch+blob |
| RSC | 10.1039/d5nr04405g | 2.1M | stealth + CDN + fetch+blob |

## 附录：各出版商 URL 模式

### Wiley

```
Landing:  https://advanced.onlinelibrary.wiley.com/doi/{DOI}
PDF:      https://advanced.onlinelibrary.wiley.com/doi/pdfdirect/{DOI}?download=true
```

### Springer Nature

```
Landing:  https://doi.org/{DOI}  → 重定向到 https://www.nature.com/articles/{article_id}
PDF:      https://www.nature.com/articles/{article_id}.pdf
```

### MDPI

```
Landing:  https://doi.org/{DOI}  → 重定向到 https://www.mdpi.com/{ISSN}/{vol}/{issue}/{article}
PDF:      https://www.mdpi.com/{ISSN}/{vol}/{issue}/{article}/pdf?version={timestamp}
```

### IEEE

```
Landing:  https://doi.org/{DOI}  → 重定向到 https://ieeexplore.ieee.org/document/{arnumber}
PDF:      https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={arnumber}
```

### ACS

```
Landing:  https://pubs.acs.org/doi/{DOI}
PDF:      https://pubs.acs.org/doi/pdf/{DOI}
```

### Elsevier

```
Landing:  https://doi.org/{DOI}  → 重定向到 https://www.sciencedirect.com/science/article/pii/{PII}
PDF:      https://www.sciencedirect.com/science/article/pii/{PII}/pdfft?md5={hash}&pid=1-s2.0-{PII}-main.pdf
          → ⚠️ 依赖机构授权！无 CARSI/Shibboleth 登录态时返回 403 或 "Get Access" 页面
          → 有授权时：JS 驱动跳转到 https://pdf.sciencedirectassets.com/{path}/main.pdf?X-Amz-Signature=...
          → 无授权时：不生成 CDN 签名 URL，等再久也不会跳转
          → 正确流程: 先检查权限 → 无权限则 CARSI 登录 → 有权限后导航 pdfft → 轮询等 CDN → fetch
```

### RSC

```
Landing:  https://doi.org/{DOI}  → 重定向到 https://pubs.rsc.org/en/content/articlelanding/{year}/{journal}/{article_code}
PDF:      https://pubs.rsc.org/en/content/articlepdf/{year}/{journal}/{article_code}
          → 导航后重定向到 https://rscj.silverchair-cdn.com/rscj/content_public/journal/{journal}/{vol}/{issue}/10.1039_{article_code}/1/{article_code}.pdf
```

**RSC 年份映射**（从 DOI 文章代码提取）：

| 文章代码前缀 | 年份 | 示例 |
|-------------|------|------|
| d0 | 2020 | d0nr04405g → 2020, Nanoscale |
| d1 | 2021 | d1ma00987k → 2021, Materials Advances |
| d2 | 2022 | d2ma00987k → 2022, Materials Advances |
| d3 | 2023 | d3nr04405g → 2023, Nanoscale |
| d4 | 2024 | d4tc01779j → 2024, Journal of Materials Chemistry C |
| d5 | 2025 | d5nr04405g → 2025, Nanoscale |
| d6 | 2026 | d6nr04405g → 2026, Nanoscale |
