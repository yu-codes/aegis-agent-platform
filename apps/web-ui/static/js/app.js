/**
 * Aegis Agent Platform - Main Application
 */

class AegisApp {
    constructor() {
        this.sessions = [];
        this.tools = [];
        this.isInitialized = false;
    }

    /**
     * Initialize application
     */
    async init() {
        console.log('Initializing Aegis Agent Platform...');
        
        // Apply saved theme
        this.applyTheme();
        
        // Apply font size
        this.applyFontSize();
        
        // Bind event listeners
        this.bindEvents();
        
        // Initialize chat
        await chat.init();
        
        // Check connection
        await this.checkConnection();
        
        // Load tools
        await this.loadTools();
        
        // Load sessions
        await this.loadSessions();
        
        // Load settings into modal
        this.loadSettingsModal();
        
        this.isInitialized = true;
        console.log('Aegis Agent Platform initialized');
    }

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Chat input
        const chatInput = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');

        if (!chatInput || !sendBtn) {
            console.error('Chat input elements not found!', {
                chatInput: !!chatInput,
                sendBtn: !!sendBtn
            });
            return;
        }

        chatInput.addEventListener('input', () => {
            Utils.autoResize(chatInput);
            sendBtn.disabled = !chatInput.value.trim();
        });

        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        sendBtn.addEventListener('click', () => this.handleSendMessage());

        // New chat button
        document.getElementById('new-chat-btn').addEventListener('click', () => {
            this.handleNewChat();
        });

        // Clear chat button
        document.getElementById('clear-chat-btn').addEventListener('click', () => {
            this.handleClearChat();
        });

        // Export button
        document.getElementById('export-btn').addEventListener('click', () => {
            chat.exportChat();
        });

        // Settings
        document.getElementById('settings-btn').addEventListener('click', () => {
            this.openSettingsModal();
        });

        document.getElementById('settings-close').addEventListener('click', () => {
            this.closeSettingsModal();
        });

        document.getElementById('settings-save').addEventListener('click', () => {
            this.saveSettings();
        });

        document.getElementById('settings-reset').addEventListener('click', () => {
            this.resetSettings();
        });

        // Modal backdrop click
        document.querySelector('#settings-modal .modal-backdrop').addEventListener('click', () => {
            this.closeSettingsModal();
        });

        // Temperature slider
        document.getElementById('temperature-slider').addEventListener('input', (e) => {
            document.getElementById('temp-value').textContent = e.target.value;
        });

        // Stream toggle
        document.getElementById('stream-toggle').addEventListener('change', (e) => {
            chat.settings.streamEnabled = e.target.checked;
        });

        // Tools toggle
        document.getElementById('tools-toggle').addEventListener('change', (e) => {
            chat.settings.toolsEnabled = e.target.checked;
        });

        // Quick actions
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.dataset.prompt;
                if (prompt) {
                    document.getElementById('chat-input').value = prompt;
                    document.getElementById('send-btn').disabled = false;
                }
            });
        });

        // Mobile menu
        document.getElementById('mobile-menu-btn').addEventListener('click', () => {
            this.toggleSidebar();
        });

        // Theme select
        document.getElementById('theme-select').addEventListener('change', (e) => {
            Utils.storage.set('aegis-theme', e.target.value);
            this.applyTheme();
        });

        // Font size select
        document.getElementById('font-size-select').addEventListener('change', (e) => {
            Utils.storage.set('aegis-font-size', e.target.value);
            this.applyFontSize();
        });
    }

    /**
     * Handle send message
     */
    async handleSendMessage() {
        const input = document.getElementById('chat-input');
        const message = input.value.trim();
        
        if (!message) return;
        
        input.value = '';
        input.style.height = 'auto';
        document.getElementById('send-btn').disabled = true;
        
        await chat.sendMessage(message);
        
        // Refresh sessions
        await this.loadSessions();
    }

    /**
     * Handle new chat
     */
    async handleNewChat() {
        await chat.newSession();
        this.setActiveSession(null);
        await this.loadSessions();
    }

    /**
     * Handle clear chat
     */
    async handleClearChat() {
        if (confirm('確定要清除目前對話嗎？')) {
            await chat.clearChat();
        }
    }

    /**
     * Check connection to API
     */
    async checkConnection() {
        const statusIndicator = document.getElementById('status-indicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');

        try {
            const health = await api.readinessCheck();
            
            statusDot.classList.remove('error');
            statusDot.classList.add('connected');
            statusText.textContent = '已連線';
            
            return true;
        } catch (error) {
            statusDot.classList.remove('connected');
            statusDot.classList.add('error');
            statusText.textContent = '連線失敗';
            
            Utils.toast('無法連接到 API 伺服器', 'error');
            return false;
        }
    }

    /**
     * Load tools list
     */
    async loadTools() {
        const toolsList = document.getElementById('tools-list');
        
        try {
            const result = await api.listTools();
            this.tools = result.tools || [];
            
            toolsList.innerHTML = this.tools.map(tool => `
                <span class="tool-tag" title="${Utils.escapeHtml(tool.description)}">
                    <i class="fas fa-wrench"></i>
                    ${Utils.escapeHtml(tool.name)}
                </span>
            `).join('');
            
        } catch (error) {
            toolsList.innerHTML = '<span class="tool-tag">載入失敗</span>';
        }
    }

    /**
     * Load sessions list
     */
    async loadSessions() {
        const sessionList = document.getElementById('session-list');
        
        try {
            const result = await api.listSessions();
            this.sessions = result.sessions || [];
            
            if (this.sessions.length === 0) {
                sessionList.innerHTML = `
                    <div class="empty-sessions">
                        <span>尚無對話歷史</span>
                    </div>
                `;
                return;
            }
            
            sessionList.innerHTML = this.sessions.map(session => `
                <div class="session-item ${session.session_id === chat.sessionId ? 'active' : ''}" 
                     data-session-id="${session.session_id}">
                    <i class="fas fa-message"></i>
                    <span class="session-title">${Utils.formatDate(new Date(session.created_at))} - ${session.message_count} 則訊息</span>
                    <button class="btn-icon session-delete" title="刪除">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            `).join('');
            
            // Bind session click events
            sessionList.querySelectorAll('.session-item').forEach(item => {
                item.addEventListener('click', (e) => {
                    if (!e.target.closest('.session-delete')) {
                        const sessionId = item.dataset.sessionId;
                        this.handleLoadSession(sessionId);
                    }
                });
                
                item.querySelector('.session-delete').addEventListener('click', (e) => {
                    e.stopPropagation();
                    const sessionId = item.dataset.sessionId;
                    this.handleDeleteSession(sessionId);
                });
            });
            
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    /**
     * Handle load session
     */
    async handleLoadSession(sessionId) {
        const success = await chat.loadSession(sessionId);
        if (success) {
            this.setActiveSession(sessionId);
            // Close sidebar on mobile
            this.closeSidebar();
        }
    }

    /**
     * Handle delete session
     */
    async handleDeleteSession(sessionId) {
        if (!confirm('確定要刪除此對話嗎？')) return;
        
        try {
            await api.deleteSession(sessionId);
            Utils.toast('對話已刪除', 'success');
            
            // If deleted current session, create new one
            if (sessionId === chat.sessionId) {
                await chat.newSession();
            }
            
            await this.loadSessions();
        } catch (error) {
            Utils.toast('刪除失敗', 'error');
        }
    }

    /**
     * Set active session in UI
     */
    setActiveSession(sessionId) {
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.toggle('active', item.dataset.sessionId === sessionId);
        });
    }

    /**
     * Toggle sidebar (mobile)
     */
    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.toggle('open');
    }

    /**
     * Close sidebar (mobile)
     */
    closeSidebar() {
        const sidebar = document.getElementById('sidebar');
        sidebar.classList.remove('open');
    }

    /**
     * Open settings modal
     */
    openSettingsModal() {
        document.getElementById('settings-modal').classList.add('active');
    }

    /**
     * Close settings modal
     */
    closeSettingsModal() {
        document.getElementById('settings-modal').classList.remove('active');
    }

    /**
     * Load settings into modal
     */
    loadSettingsModal() {
        const settings = chat.settings;
        
        document.getElementById('model-select').value = settings.model || 'stub';
        document.getElementById('temperature-slider').value = settings.temperature || 0.7;
        document.getElementById('temp-value').textContent = settings.temperature || 0.7;
        document.getElementById('max-tokens-input').value = settings.maxTokens || 4096;
        document.getElementById('stream-toggle').checked = settings.streamEnabled !== false;
        document.getElementById('tools-toggle').checked = settings.toolsEnabled !== false;
        
        document.getElementById('theme-select').value = Utils.storage.get('aegis-theme', 'dark');
        document.getElementById('font-size-select').value = Utils.storage.get('aegis-font-size', 'medium');
        document.getElementById('api-url-input').value = api.baseUrl;
    }

    /**
     * Save settings from modal
     */
    saveSettings() {
        chat.settings.model = document.getElementById('model-select').value;
        chat.settings.temperature = parseFloat(document.getElementById('temperature-slider').value);
        chat.settings.maxTokens = parseInt(document.getElementById('max-tokens-input').value);
        chat.settings.streamEnabled = document.getElementById('stream-toggle').checked;
        chat.settings.toolsEnabled = document.getElementById('tools-toggle').checked;
        
        chat.saveSettings();
        chat.updateModelBadge(chat.settings.model);
        
        // Update API URL if changed
        const newUrl = document.getElementById('api-url-input').value.trim();
        if (newUrl && newUrl !== api.baseUrl) {
            api.setBaseUrl(newUrl);
            Utils.storage.set('aegis-api-url', newUrl);
            this.checkConnection();
        }
        
        this.closeSettingsModal();
        Utils.toast('設定已儲存', 'success');
    }

    /**
     * Reset settings to defaults
     */
    resetSettings() {
        chat.settings = {
            streamEnabled: true,
            toolsEnabled: true,
            model: 'stub',
            temperature: 0.7,
            maxTokens: 4096,
        };
        
        this.loadSettingsModal();
        Utils.toast('設定已重置', 'info');
    }

    /**
     * Apply theme
     */
    applyTheme() {
        const theme = Utils.storage.get('aegis-theme', 'dark');
        
        if (theme === 'auto') {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.documentElement.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
        } else {
            document.documentElement.setAttribute('data-theme', theme);
        }
    }

    /**
     * Apply font size
     */
    applyFontSize() {
        const fontSize = Utils.storage.get('aegis-font-size', 'medium');
        document.documentElement.setAttribute('data-font-size', fontSize);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOMContentLoaded event fired');
    try {
        window.app = new AegisApp();
        window.app.init().catch(error => {
            console.error('Initialization failed:', error);
            Utils.toast('應用程式初始化失敗: ' + error.message, 'error');
        });
    } catch (error) {
        console.error('Failed to create app:', error);
        alert('應用程式初始化失敗: ' + error.message);
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AegisApp;
}
