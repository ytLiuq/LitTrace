# Paper PDF Downloader — 学术论文通用下载技能

## 概述

一套方案，覆盖七家主流学术出版商的论文 PDF 下载：**Wiley、Springer Nature、MDPI、IEEE、ACS、Elsevier、RSC**。

核心设计：**两个决策点 × 三种下载手段，三步递进**——输入 DOI 即可自动完成下载，无需手动指定出版商。

## 触发场景

当用户提到以下内容时使用本技能：
- 下载论文 / 下载 PDF / 下载学术论文
- 获取全文 / 获取 PDF
- 论文下载
- 某个 DOI 对应的论文
- 指定出版商的论文下载（Wiley、Springer Nature、MDPI、IEEE、ACS、Elsevier、RSC）

关键触发词：下载论文、下载 PDF、论文 PDF、获取全文、DOI 下载、学术论文下载、paper download、fetch PDF

## 前置条件

1. **CDP 浏览器**：需启动带远程调试端口的 Chrome
   ```bash
   google-chrome --remote-debugging-port=9222
   ```
   验证：`curl -s http://127.0.0.1:9222/json/version` 返回 JSON

2. **Python 依赖**：
   ```bash
   pip install requests websocket-client
   ```

3. **Unpaywall 邮箱**：必须使用真实域名的邮箱（如 `research@sjtu.edu.cn`），`test@example.com` 会被拒绝（HTTP 422）

4. **机构认证**：部分出版商（ACS、IEEE 非 OA 论文）需要 CARSI/Shibboleth 机构认证，脚本会提示用户在浏览器中手动完成

## 脚本位置

```
{SKILL_DIR}/scripts/universal_paper_downloader.py
```

## 用法

```bash
# 基本用法
python3 scripts/universal_paper_downloader.py <DOI>

# 指定输出路径
python3 scripts/universal_paper_downloader.py 10.1039/d5nr04405g -o ~/Desktop/paper.pdf

# 指定 CDP 地址和邮箱
python3 scripts/universal_paper_downloader.py 10.1021/acs.nano.5c21970 \
    --cdp http://127.0.0.1:9222 --email research@sjtu.edu.cn
```

## 统一决策流程

```
输入: DOI
  │
  ├─ Step 1: 从 DOI 前缀自动识别出版商
  │   10.1002→Wiley, 10.1038→Springer Nature, 10.3390→MDPI,
  │   10.1109→IEEE, 10.1021→ACS, 10.1016→Elsevier, 10.1039→RSC
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
  │   ├─ 导航到 landing page（doi.org 自动重定向到正确页面）
  │   │   ├─ Cloudflare managed challenge → stealth 自动通过
  │   │   ├─ Cloudflare Captcha → 提示用户手动验证
  │   │   └─ 无 Cloudflare → 继续
  │   ├─ 检测是否需要 CARSI 登录 → 提示用户完成
  │   │
  │   └─ 下载 PDF（三种子方法，按出版商自动选择）:
  │       ├─ 方法 A: CDP setDownloadBehavior + goto
  │       │   适用: Wiley
  │       │   原理: 设置下载路径后导航到 PDF URL，浏览器自动触发下载
  │       │
  │       ├─ 方法 B: fetch + blob + base64 回传
  │       │   适用: MDPI、IEEE、ACS、RSC、Elsevier
  │       │   原理: 在页面内用 JS fetch PDF blob，转 base64 回传 Python 写盘
  │       │   注意: 同域用相对路径避免 CORS
  │       │   注意: RSC 需先导航到 PDF URL 获取 silverchair CDN 重定向地址
  │       │   注意: Elsevier 需先导航到 pdfft URL，等待重定向到
  │       │         pdf.sciencedirectassets.com CDN 后再 fetch（同域）
  │       │
  │       └─ 方法 C: <a download> 触发浏览器原生下载
  │           适用: fetch 失败时的 fallback
  │
  └─ 输出: PDF 文件 + 使用的下载方法
```

## 七家出版商路径速查

| 出版商 | DOI 前缀 | 典型路径 | 需要用户交互 |
|--------|---------|---------|-------------|
| Wiley | 10.1002 | stealth → setDownloadBehavior + goto | 否 |
| Springer Nature | 10.1038 | Unpaywall OA → fetch+blob（或 curl 直下） | 否 |
| MDPI | 10.3390 | stealth → fetch+blob（相对路径） | 否 |
| IEEE | 10.1109 | Unpaywall → repository（arXiv）curl 直下；若无 OA 则 stealth + fetch+blob | 非 OA 需 CARSI |
| ACS | 10.1021 | stealth → Cloudflare 自动通过 → fetch+blob | Cloudflare 严格时需手动验证 |
| Elsevier | 10.1016 | stealth → 提取 pdfft URL → 导航到 pdfft → CDN 重定向 → fetch+blob | 否 |
| RSC | 10.1039 | stealth → landing → silverchair CDN → fetch+blob | 否 |

> 关键洞察：
> - Elsevier 的 pdfft URL 返回 HTML（JS 跳转页）而非 PDF，需先导航到 pdfft 让浏览器重定向到 `pdf.sciencedirectassets.com` CDN，再从 CDN 同域 fetch+blob
> - ACS 的 Cloudflare managed challenge 可被 stealth 8 项覆盖自动通过，但偶尔严格时需用户手动验证（脚本自动等待 120 秒）
> - IEEE 的 OA 论文通常有 arXiv 副本

## stealth 注入脚本（8 项覆盖）

以下 JavaScript 在导航到目标页面前通过 `Page.addScriptToEvaluateOnNewDocument` 注入，用于绕过 Cloudflare managed challenge：

```javascript
// 1. navigator.webdriver → undefined
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
// 2. window.chrome 伪造
window.chrome = { runtime: {} };
// 3. permissions.query 覆盖
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
    parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
);
// 4. navigator.plugins
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
// 5. navigator.languages
Object.defineProperty(navigator, 'languages', { get: () => ['zh-CN', 'zh', 'en'] });
// 6. navigator.connection
if (navigator.connection === undefined) {
    Object.defineProperty(navigator, 'connection', {
        get: () => ({ effectiveType: '4g', rtt: 50, downlink: 10 })
    });
}
// 7. WebGL vendor 覆盖
const getParameter = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(parameter) {
    if (parameter === 37445) return 'Intel Inc.';
    if (parameter === 37446) return 'Intel Iris OpenGL Engine';
    return getParameter.apply(this, arguments);
};
// 8. cdc_ 变量清理
for (const key of Object.keys(window)) {
    if (key.startsWith('cdc_') || key.startsWith('$cdc_')) delete window[key];
}
```

**适用性**：
- 对 ACS、RSC 的 Cloudflare managed challenge 有效（自动通过）
- 对 Elsevier 的 Cloudflare Captcha 无效（需 Unpaywall 绕道 repository）

## CDPBrowser 关键 API

脚本封装了 `CDPBrowser` 类，核心方法：

| 方法 | 用途 |
|------|------|
| `connect_new_tab()` | 创建新标签页并连接 WebSocket |
| `inject_stealth()` | 注入 8 项 stealth 覆盖 |
| `set_download_path(path)` | 设置浏览器下载目录 |
| `navigate(url, wait)` | 导航到 URL 并等待 |
| `is_cloudflare_challenge()` | 检测是否在 Cloudflare 验证页 |
| `wait_for_cloudflare(max_wait)` | 等待 Cloudflare 自动通过 |
| `fetch_blob_and_save(pdf_url, output_path)` | 同域 fetch PDF + base64 回传写盘 |
| `trigger_download_via_anchor(pdf_url, filename)` | `<a download>` 触发浏览器下载 |
| `send(method, params)` | 底层 CDP 命令（带 3 次自动重连） |

## 工作流步骤（Agent 执行指南）

### 1. 确认输入

用户提供 DOI（如 `10.1039/d5nr04405g`）。如果用户只给了论文标题，先用 CrossRef API 查 DOI：
```bash
curl -s "https://api.crossref.org/works?query.bibliographic=<标题>&rows=1&select=DOI,title" | python3 -c "
import sys, json; d=json.load(sys.stdin); print(d['message']['items'][0]['DOI'])"
```

### 2. 检查前置条件

```bash
# 检查 CDP 浏览器
curl -s http://127.0.0.1:19222/json/version | python3 -c "import sys,json; print('OK')"

# 检查 Python 依赖
python3 -c "import requests, websocket; print('OK')"
```

如果 CDP 浏览器未运行，提示用户启动：
```bash
google-chrome --remote-debugging-port=9222
```

### 3. 执行脚本

```bash
python3 {SKILL_DIR}/scripts/universal_paper_downloader.py <DOI> \
    -o <输出路径> \
    --cdp http://127.0.0.1:19222 \
    --email research@sjtu.edu.cn
```

### 4. 处理用户交互

脚本在以下情况会暂停等待用户操作（非交互模式自动等待 60 秒）：
- **Cloudflare Captcha**：ACS/Elsevier 的 Captcha 无法自动通过时，需用户在浏览器中勾选验证。脚本会先自动等待 60 秒，再提示用户手动操作，用户完成后最多再等 120 秒
- **CARSI 登录**：ACS/IEEE 的非 OA 论文需要机构认证，需用户在浏览器中完成 SJTU CARSI 登录

### 5. 验证结果

```bash
# 检查文件是否是真实 PDF
file <输出路径>
# 应输出 "PDF document, version X.X"
```

## 故障排查

| 症状 | 原因 | 解决方案 |
|------|------|---------|
| `CDP 浏览器未运行` | Chrome 未启动或端口不对 | 启动 `google-chrome --remote-debugging-port=9222`，用 `--cdp` 指定正确端口 |
| `Unpaywall 拒绝请求` | 邮箱无效 | 用真实域名邮箱 `--email research@sjtu.edu.cn` |
| `下载到的是 HTML 而非 PDF` | Cloudflare 拦截 curl | 正常，脚本会自动进入浏览器方案 |
| `fetch 失败: Failed to fetch` | CORS 跨域 | 脚本自动用相对路径重试；如果仍失败检查是否在同域 |
| `WebSocket 断连` | 导航后连接丢失 | 脚本有 3 次自动重连机制 |
| `Cloudflare Captcha 无法自动通过` | ACS/Elsevier 的 Captcha | 需用户在浏览器中手动完成验证，脚本自动等待最多 120 秒 |
| `无法找到 PDF 下载链接` | 页面结构变化或需要登录 | 检查是否需要 CARSI 登录；手动在浏览器中找到 PDF 链接 |
| IEEE `javascript:void()` | PDF 按钮是 JS 触发 | 脚本从 HTML 提取 `arnumber` 构造 stampPDF URL |
| MDPI `setDownloadBehavior` 覆盖了脚本文件 | 下载路径与脚本同目录 | 用 `-o /tmp/output.pdf` 指定输出到其他目录 |
| Elsevier `pdfft 返回 HTML` | pdfft 是 JS 跳转页非直接 PDF | 脚本自动导航到 pdfft URL，等待重定向到 `pdf.sciencedirectassets.com` CDN 后再 fetch+blob |

## 已知限制

1. **ACS Cloudflare 波动**：stealth 8 项覆盖通常能自动通过 Cloudflare managed challenge，但 Cloudflare 严格时（约 20% 概率）无法自动通过，需用户在浏览器中手动勾选验证。脚本会先自动等待 60 秒，再提示用户操作，完成后最多再等 120 秒
2. **IEEE 非 OA 论文**：需要机构 IP 或 CARSI 认证，脚本会提示用户完成登录
3. **CARSI 登录态不持久**：关闭浏览器后失效，下次使用需重新登录
4. **Unpaywall 覆盖延迟**：最新发表的论文可能尚未被 Unpaywall 索引（通常 1-2 周延迟）
5. **Elsevier PII 获取**：脚本从 doi.org 重定向获取 ScienceDirect 的 PII 号，确保导航到正确的文章页面

## 测试验证

本技能已于 2026-07-10 完成两轮七家出版商全量测试，成功率 7/7：

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
