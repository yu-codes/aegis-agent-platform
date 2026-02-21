# Aegis Web UI 訪問指南

## 🌐 Web 用戶界面

Aegis Agent Platform 提供了完整的 Web UI 供用戶互動使用。

### 訪問地址

| 版本 | 地址 | 說明 |
|------|------|------|
| **標準版** | `http://localhost:8080/ui` | 功能完整，支援深淺色主題、設定面板等 |
| **簡單版** | `http://localhost:8080/simple` | 自包含 HTML，適合調試和基礎測試 |
| **測試頁** | `http://localhost:8080/test` | API 連線測試工具 |

### 連接故障排解

#### 問題：無法在瀏覽器中訪問 localhost:8080

**可能原因**：主機和 Docker 容器之間的 IPv6/IPv4 連接問題

**解決方案**：

1. **使用簡單版本** (推薦)
   ```bash
   # 訪問簡單版本
   http://localhost:8080/simple
   ```

2. **使用容器 IP 地址**
   ```bash
   # 查看容器 IP
   docker inspect aegis-api | grep IPAddress
   
   # 訪問
   http://192.168.0.3:8080/ui  # 根據實際 IP 調整
   ```

3. **從容器內測試**
   ```bash
   docker exec aegis-api curl -s http://localhost:8080/ui
   docker exec aegis-api curl -s http://localhost:8080/api/v1/tools
   ```

4. **檢查 API 連線**
   ```bash
   # 測試 API 健康狀態
   http://localhost:8080/health
   http://localhost:8080/health/ready
   ```

### 功能說明

#### 標準版 UI (/ui)
- 💬 **對話管理**：新增、載入、刪除對話工作階段
- ⚙️ **設定面板**：
  - 選擇 AI 模型（Stub, GPT-4o, Claude 等）
  - 調整溫度和最大 tokens
  - 啟用/禁用串流和工具
- 🎨 **主題切換**：深色/淺色模式
- 📱 **響應式設計**：支援手機和平板

#### 簡單版 UI (/simple)
- 🔌 **API 測試**：一鍵測試 API 連線
- 💬 **基礎對話**：收發訊息
- 🔨 **工具列表**：查看可用工具
- 📊 **狀態顯示**：實時連線狀態

### API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/health` | GET | 健康狀態檢查 |
| `/health/ready` | GET | 就緒狀態檢查 |
| `/api/v1/chat` | POST | 發送聊天訊息 |
| `/api/v1/sessions` | POST/GET | 建立/列出工作階段 |
| `/api/v1/tools` | GET | 取得可用工具列表 |

### 調試技巧

#### 1. 查看 API 容器日誌
```bash
docker logs aegis-api -f --tail 50
```

#### 2. 測試各個靜態資源
```bash
# CSS
curl -I http://localhost:8080/static/css/main.css

# JavaScript
curl -I http://localhost:8080/static/js/app.js
```

#### 3. 測試 API 端點
```bash
# 建立工作階段
curl -X POST http://localhost:8080/api/v1/sessions

# 發送訊息
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "stream": false}'
```

### 常見問題

**Q: 為什麼無法在瀏覽器中看到網頁？**  
A: 可能是 IPv6 連接問題。嘗試使用簡單版本 (/simple) 或查看上面的故障排解方案。

**Q: API 連線失敗怎麼辦？**  
A: 執行以下命令檢查：
```bash
docker ps | grep aegis  # 檢查容器是否運行
docker logs aegis-api | tail -20  # 查看最近的日誌
curl http://localhost:8080/health  # 測試 API
```

**Q: 如何修改 API 連線地址？**  
A: 在標準版 UI 的設定面板中，可以修改 API URL。

## 🚀 快速開始

1. **啟動服務**
   ```bash
   docker compose up -d aegis redis
   ```

2. **訪問 Web UI**
   - 標準版：http://localhost:8080/ui
   - 簡單版：http://localhost:8080/simple

3. **測試 API**
   按簡單版中的"🔌 測試 API"按鈕

## 📝 建築文件

| 檔案 | 說明 |
|------|------|
| [apps/web-ui/static/index.html](../apps/web-ui/static/index.html) | 標準版 UI HTML |
| [apps/web-ui/static/simple.html](../apps/web-ui/static/simple.html) | 簡單版 UI HTML |
| [apps/web-ui/static/css/main.css](../apps/web-ui/static/css/main.css) | UI 樣式表 |
| [apps/web-ui/static/js/app.js](../apps/web-ui/static/js/app.js) | 主應用程式 |
| [apps/web-ui/static/js/api.js](../apps/web-ui/static/js/api.js) | API 客戶端 |
| [apps/web-ui/static/js/chat.js](../apps/web-ui/static/js/chat.js) | 聊天管理器 |
| [apps/web-ui/static/js/utils.js](../apps/web-ui/static/js/utils.js) | 工具函式 |

## 端口配置

- **API 伺服器**：8080 (內部) → 8080 (主機)
- **Redis**：6379
- **開發伺服器**：8001 (profile: dev)
- **離線模式**：8002 (profile: offline)

更新：所有端口已從 8000 改為 8080（主 API）

---

需要幫助？檢查 [Docker 日誌](#1-查看-api-容器日誌) 或查閱 [API 文件](/docs)
