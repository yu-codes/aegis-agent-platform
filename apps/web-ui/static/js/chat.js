/**
 * Aegis Agent Platform - Chat Manager
 */

class ChatManager {
    constructor() {
        this.sessionId = null;
        this.messages = [];
        this.isStreaming = false;
        this.currentMessageElement = null;
        
        // Settings
        this.settings = {
            streamEnabled: true,
            toolsEnabled: true,
            model: 'stub',
            temperature: 0.7,
            maxTokens: 4096,
        };
    }

    /**
     * Initialize chat manager
     */
    async init() {
        // Load settings
        this.loadSettings();
        
        // Start new session
        await this.newSession();
    }

    /**
     * Load settings from storage
     */
    loadSettings() {
        const saved = Utils.storage.get('aegis-settings');
        if (saved) {
            this.settings = { ...this.settings, ...saved };
        }
    }

    /**
     * Save settings to storage
     */
    saveSettings() {
        Utils.storage.set('aegis-settings', this.settings);
    }

    /**
     * Create new session
     */
    async newSession() {
        try {
            const result = await api.createSession();
            this.sessionId = result.session_id;
            this.messages = [];
            this.clearMessageDisplay();
            this.showWelcome();
            this.updateChatTitle('新對話');
            return this.sessionId;
        } catch (error) {
            Utils.toast('建立會話失敗: ' + error.message, 'error');
            // Generate local session ID as fallback
            this.sessionId = Utils.generateId();
            return this.sessionId;
        }
    }

    /**
     * Load session
     */
    async loadSession(sessionId) {
        try {
            const result = await api.getSession(sessionId);
            this.sessionId = sessionId;
            this.messages = result.messages || [];
            this.renderAllMessages();
            
            // Update title from first user message
            const firstUserMsg = this.messages.find(m => m.role === 'user');
            this.updateChatTitle(firstUserMsg ? Utils.truncate(firstUserMsg.content, 30) : '已載入對話');
            
            return true;
        } catch (error) {
            Utils.toast('載入會話失敗', 'error');
            return false;
        }
    }

    /**
     * Send message
     */
    async sendMessage(content) {
        if (!content.trim() || this.isStreaming) return;

        // Hide welcome message
        this.hideWelcome();

        // Add user message
        const userMessage = {
            role: 'user',
            content: content.trim(),
            timestamp: new Date(),
        };
        this.messages.push(userMessage);
        this.renderMessage(userMessage);

        // Update title if first message
        if (this.messages.length === 1) {
            this.updateChatTitle(Utils.truncate(content, 30));
        }

        // Show typing indicator
        this.showTyping();

        try {
            if (this.settings.streamEnabled) {
                await this.sendStreamingMessage(content);
            } else {
                await this.sendNonStreamingMessage(content);
            }
        } catch (error) {
            this.hideTyping();
            Utils.toast('發送訊息失敗: ' + error.message, 'error');
            
            // Add error message
            const errorMessage = {
                role: 'assistant',
                content: `抱歉，發生錯誤: ${error.message}`,
                timestamp: new Date(),
                error: true,
            };
            this.messages.push(errorMessage);
            this.renderMessage(errorMessage);
        }
    }

    /**
     * Send message with streaming
     */
    async sendStreamingMessage(content) {
        this.isStreaming = true;
        
        const assistantMessage = {
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            toolsUsed: [],
        };
        this.messages.push(assistantMessage);
        
        // Create message element for streaming
        this.currentMessageElement = this.createMessageElement(assistantMessage);
        this.hideTyping();
        document.getElementById('chat-messages').appendChild(this.currentMessageElement);
        
        const bubbleEl = this.currentMessageElement.querySelector('.message-bubble');

        try {
            const stream = api.streamMessage(content, {
                sessionId: this.sessionId,
                model: this.settings.model,
                temperature: this.settings.temperature,
                maxTokens: this.settings.maxTokens,
                toolsEnabled: this.settings.toolsEnabled,
            });

            for await (const data of stream) {
                if (data.type === 'text') {
                    assistantMessage.content += data.content;
                    bubbleEl.innerHTML = Utils.parseMarkdown(assistantMessage.content);
                    this.scrollToBottom();
                } else if (data.type === 'done') {
                    this.sessionId = data.session_id || this.sessionId;
                    assistantMessage.toolsUsed = data.tools_used || [];
                    
                    // Add tools used badge
                    if (assistantMessage.toolsUsed.length > 0) {
                        this.addToolsBadge(this.currentMessageElement, assistantMessage.toolsUsed);
                    }
                } else if (data.type === 'error') {
                    throw new Error(data.message);
                }
            }
        } finally {
            this.isStreaming = false;
            this.currentMessageElement = null;
        }
    }

    /**
     * Send message without streaming
     */
    async sendNonStreamingMessage(content) {
        try {
            const result = await api.sendMessage(content, {
                sessionId: this.sessionId,
                model: this.settings.model,
                temperature: this.settings.temperature,
                maxTokens: this.settings.maxTokens,
                toolsEnabled: this.settings.toolsEnabled,
            });

            this.hideTyping();
            this.sessionId = result.session_id || this.sessionId;

            const assistantMessage = {
                role: 'assistant',
                content: result.message,
                timestamp: new Date(),
                toolsUsed: result.tools_used || [],
                metadata: result.metadata,
            };
            this.messages.push(assistantMessage);
            this.renderMessage(assistantMessage);

        } catch (error) {
            this.hideTyping();
            throw error;
        }
    }

    /**
     * Render single message
     */
    renderMessage(message) {
        const element = this.createMessageElement(message);
        document.getElementById('chat-messages').appendChild(element);
        this.scrollToBottom();
    }

    /**
     * Create message DOM element
     */
    createMessageElement(message) {
        const div = document.createElement('div');
        div.className = `message ${message.role}`;
        
        const isUser = message.role === 'user';
        const avatarIcon = isUser ? 'fa-user' : 'fa-robot';
        
        div.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${avatarIcon}"></i>
            </div>
            <div class="message-content">
                <div class="message-bubble">
                    ${Utils.parseMarkdown(message.content)}
                </div>
                <div class="message-meta">
                    <span>${Utils.formatTime(message.timestamp)}</span>
                    ${message.metadata?.model ? `<span>• ${message.metadata.model}</span>` : ''}
                    ${message.metadata?.tokens_used ? `<span>• ${message.metadata.tokens_used} tokens</span>` : ''}
                </div>
                ${message.toolsUsed?.length > 0 ? this.createToolsBadgeHtml(message.toolsUsed) : ''}
            </div>
        `;
        
        return div;
    }

    /**
     * Create tools badge HTML
     */
    createToolsBadgeHtml(tools) {
        const badges = tools.map(tool => 
            `<span class="tool-used-badge"><i class="fas fa-wrench"></i>${tool}</span>`
        ).join('');
        return `<div class="tools-used">${badges}</div>`;
    }

    /**
     * Add tools badge to existing message
     */
    addToolsBadge(element, tools) {
        const content = element.querySelector('.message-content');
        const html = this.createToolsBadgeHtml(tools);
        content.insertAdjacentHTML('beforeend', html);
    }

    /**
     * Render all messages
     */
    renderAllMessages() {
        this.clearMessageDisplay();
        if (this.messages.length === 0) {
            this.showWelcome();
        } else {
            this.hideWelcome();
            this.messages.forEach(msg => this.renderMessage(msg));
        }
    }

    /**
     * Clear message display
     */
    clearMessageDisplay() {
        const container = document.getElementById('chat-messages');
        const welcome = document.getElementById('welcome-message');
        container.innerHTML = '';
        if (welcome) {
            container.appendChild(welcome);
        }
    }

    /**
     * Clear current chat
     */
    async clearChat() {
        if (this.sessionId) {
            try {
                await api.clearSession(this.sessionId);
            } catch {
                // Ignore error
            }
        }
        this.messages = [];
        this.clearMessageDisplay();
        this.showWelcome();
        Utils.toast('對話已清除', 'success');
    }

    /**
     * Show welcome message
     */
    showWelcome() {
        const welcome = document.getElementById('welcome-message');
        if (welcome) {
            welcome.style.display = 'flex';
        }
    }

    /**
     * Hide welcome message
     */
    hideWelcome() {
        const welcome = document.getElementById('welcome-message');
        if (welcome) {
            welcome.style.display = 'none';
        }
    }

    /**
     * Show typing indicator
     */
    showTyping() {
        let typing = document.querySelector('.typing-indicator-wrapper');
        if (!typing) {
            typing = document.createElement('div');
            typing.className = 'message assistant typing-indicator-wrapper';
            typing.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-bubble">
                        <div class="typing-indicator">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('chat-messages').appendChild(typing);
        }
        this.scrollToBottom();
    }

    /**
     * Hide typing indicator
     */
    hideTyping() {
        const typing = document.querySelector('.typing-indicator-wrapper');
        if (typing) {
            typing.remove();
        }
    }

    /**
     * Update chat title
     */
    updateChatTitle(title) {
        const titleEl = document.getElementById('chat-title');
        if (titleEl) {
            titleEl.textContent = title;
        }
    }

    /**
     * Update model badge
     */
    updateModelBadge(model) {
        const badge = document.getElementById('model-badge');
        if (badge) {
            badge.textContent = model;
        }
    }

    /**
     * Scroll to bottom of chat
     */
    scrollToBottom() {
        const container = document.getElementById('chat-messages');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    /**
     * Export chat as JSON
     */
    exportChat() {
        const exportData = {
            sessionId: this.sessionId,
            exportTime: new Date().toISOString(),
            messages: this.messages.map(m => ({
                role: m.role,
                content: m.content,
                timestamp: m.timestamp?.toISOString(),
                toolsUsed: m.toolsUsed,
            })),
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `aegis-chat-${this.sessionId || 'export'}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        Utils.toast('對話已匯出', 'success');
    }
}

// Create global instance
const chat = new ChatManager();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { ChatManager, chat };
}
