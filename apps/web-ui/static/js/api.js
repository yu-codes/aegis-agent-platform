/**
 * Aegis Agent Platform - API Client
 */

class AegisAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || window.location.origin;
    }

    /**
     * Set base URL
     */
    setBaseUrl(url) {
        this.baseUrl = url.replace(/\/$/, '');
    }

    /**
     * Make API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);
            
            if (!response.ok) {
                const error = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(error.detail || error.message || 'API Error');
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    }

    /**
     * Health check
     */
    async healthCheck() {
        return this.request('/health');
    }

    /**
     * Readiness check
     */
    async readinessCheck() {
        return this.request('/health/ready');
    }

    /**
     * Send chat message (non-streaming)
     */
    async sendMessage(message, options = {}) {
        return this.request('/api/v1/chat', {
            method: 'POST',
            body: JSON.stringify({
                message,
                session_id: options.sessionId,
                stream: false,
                model: options.model,
                temperature: options.temperature,
                max_tokens: options.maxTokens,
                tools_enabled: options.toolsEnabled ?? true,
                domain: options.domain,
            }),
        });
    }

    /**
     * Send chat message with streaming
     */
    async *streamMessage(message, options = {}) {
        const url = `${this.baseUrl}/api/v1/chat`;
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                session_id: options.sessionId,
                stream: true,
                model: options.model,
                temperature: options.temperature,
                max_tokens: options.maxTokens,
                tools_enabled: options.toolsEnabled ?? true,
                domain: options.domain,
            }),
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({ detail: response.statusText }));
            throw new Error(error.detail || 'Stream Error');
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            yield data;
                        } catch {
                            // Skip invalid JSON
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }

    /**
     * Create session
     */
    async createSession(domain = null) {
        return this.request('/api/v1/sessions', {
            method: 'POST',
            body: JSON.stringify({ domain }),
        });
    }

    /**
     * List sessions
     */
    async listSessions(limit = 20, offset = 0) {
        return this.request(`/api/v1/sessions?limit=${limit}&offset=${offset}`);
    }

    /**
     * Get session history
     */
    async getSession(sessionId) {
        return this.request(`/api/v1/sessions/${sessionId}`);
    }

    /**
     * Delete session
     */
    async deleteSession(sessionId) {
        return this.request(`/api/v1/sessions/${sessionId}`, {
            method: 'DELETE',
        });
    }

    /**
     * Clear session history
     */
    async clearSession(sessionId) {
        return this.request(`/api/v1/sessions/${sessionId}/clear`, {
            method: 'POST',
        });
    }

    /**
     * List tools
     */
    async listTools() {
        return this.request('/api/v1/tools');
    }

    /**
     * Get tool details
     */
    async getTool(toolName) {
        return this.request(`/api/v1/tools/${toolName}`);
    }

    /**
     * Call tool directly
     */
    async callTool(toolName, args, sessionId = null) {
        return this.request('/api/v1/tools/call', {
            method: 'POST',
            body: JSON.stringify({
                tool_name: toolName,
                arguments: args,
                session_id: sessionId,
            }),
        });
    }

    /**
     * Submit feedback
     */
    async submitFeedback(sessionId, messageIndex, rating, feedbackText = null) {
        return this.request('/api/v1/chat/feedback', {
            method: 'POST',
            body: JSON.stringify({
                session_id: sessionId,
                message_index: messageIndex,
                rating,
                feedback_text: feedbackText,
            }),
        });
    }

    /**
     * Get system stats
     */
    async getStats() {
        return this.request('/api/v1/admin/stats');
    }

    /**
     * Get available domains
     */
    async getDomains() {
        return this.request('/api/v1/admin/domains');
    }
}

// Create global instance
const api = new AegisAPI();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AegisAPI, api };
}
