/**
 * Qwen API 代理服务器
 * 解决浏览器 CORS 跨域问题
 * 
 * 使用方法：
 * 1. npm install express node-fetch dotenv
 * 2. node proxy-server.js
 */

import express from 'express'
import cors from 'cors'
import fetch from 'node-fetch'
import dotenv from 'dotenv'

dotenv.config({ path: '.env.development' })

const app = express()
const PORT = 3001

// 请求去重：记录最近的请求，防止重复
const recentRequests = new Map()
const REQUEST_COOLDOWN = 2000 // 2秒内相同请求视为重复

// 允许跨域
app.use(cors())
app.use(express.json())

// 请求去重中间件
app.use((req, res, next) => {
  if (req.method === 'POST' && req.path === '/api/qwen/chat') {
    const requestKey = JSON.stringify(req.body)
    const lastRequestTime = recentRequests.get(requestKey)
    const now = Date.now()
    
    if (lastRequestTime && (now - lastRequestTime) < REQUEST_COOLDOWN) {
      console.log('[代理] 拦截重复请求')
      return res.status(429).json({ 
        error: '请求过于频繁，请稍后再试' 
      })
    }
    
    recentRequests.set(requestKey, now)
    
    // 清理旧请求记录
    if (recentRequests.size > 100) {
      const oldestKey = recentRequests.keys().next().value
      recentRequests.delete(oldestKey)
    }
  }
  next()
})

// Qwen API 代理端点
app.post('/api/qwen/chat', async (req, res) => {
  const apiKey = process.env.VITE_QWEN_API_KEY
  // 固定使用阿里云API地址，不从环境变量读取（避免循环）
  const apiUrl = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
  
  if (!apiKey || apiKey === 'your-api-key-here') {
    return res.status(401).json({ 
      error: '未配置 QWEN_API_KEY，请在 .env.development 中设置' 
    })
  }

  try {
    console.log('[代理] 收到请求:', {
      model: req.body.model,
      messages: req.body.messages?.length + ' 条消息'
    })

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(req.body)
    })

    const data = await response.json()
    
    if (!response.ok) {
      console.error('[代理] API 错误:', data)
      return res.status(response.status).json(data)
    }

    console.log('[代理] 响应成功，返回内容:', {
      hasChoices: !!data.choices,
      choicesLength: data.choices?.length,
      firstMessage: data.choices?.[0]?.message?.content?.substring(0, 100)
    })
    res.json(data)
  } catch (error) {
    console.error('[代理] 请求失败:', error)
    res.status(500).json({ 
      error: '代理请求失败: ' + error.message 
    })
  }
})

// 健康检查
app.get('/health', (req, res) => {
  const apiKey = process.env.VITE_QWEN_API_KEY
  res.json({ 
    status: 'ok',
    apiKeyConfigured: !!(apiKey && apiKey !== 'your-api-key-here'),
    timestamp: new Date().toISOString()
  })
})

app.listen(PORT, () => {
  console.log(`\n✅ Qwen API 代理服务器启动成功！`)
  console.log(`📡 监听端口: http://localhost:${PORT}`)
  console.log(`🔑 API Key: ${process.env.VITE_QWEN_API_KEY ? '已配置' : '❌ 未配置'}`)
  console.log(`\n💡 使用方法:`)
  console.log(`   在前端设置: VITE_QWEN_API_URL=http://localhost:${PORT}/api/qwen/chat\n`)
})
