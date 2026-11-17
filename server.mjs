import 'dotenv/config';
import express from 'express';
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const PORT = Number(process.env.PORT || 3000);
const IS_PROD = process.env.NODE_ENV === 'production';

function createGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return null;
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function parseDataUrl(dataUrl) {
  if (typeof dataUrl !== 'string') return null;
  const match = dataUrl.match(/^data:(image\/(?:jpeg|jpg|png|webp));base64,([A-Za-z0-9+/=]+)$/i);
  if (!match) return null;
  const mimeType = match[1].toLowerCase() === 'image/jpg' ? 'image/jpeg' : match[1].toLowerCase();
  return {
    mimeType,
    data: match[2],
  };
}

function extractReadableErrorMessage(raw) {
  if (!raw) return 'Unknown server error.';
  try {
    const outer = JSON.parse(raw);
    const innerMsg = outer?.error?.message || outer?.message;
    if (typeof innerMsg === 'string') {
      try {
        const inner = JSON.parse(innerMsg);
        if (inner?.error?.message) return inner.error.message;
      } catch {
        return innerMsg;
      }
    }
  } catch {
    // Not raw JSON
  }
  return raw;
}

function formatGeminiError(error) {
  const raw = error instanceof Error ? error.message : String(error || 'Unknown error');
  const readable = extractReadableErrorMessage(raw);
  if (raw.includes('API_KEY_INVALID') || raw.includes('PERMISSION_DENIED') || raw.includes('403') || raw.includes('401')) {
    return {
      status: 401,
      message: 'Gemini API authentication failed. Please verify GEMINI_API_KEY in Settings > Secrets.',
    };
  }
  if (raw.includes('RESOURCE_EXHAUSTED') || raw.includes('429')) {
    return {
      status: 429,
      message: 'Gemini API rate limit reached. Please wait a moment and try again.',
    };
  }
  if (raw.includes('503') || raw.includes('UNAVAILABLE')) {
    return {
      status: 503,
      message: 'The AI service is temporarily experiencing high demand. Please try again in a moment.',
    };
  }
  return {
    status: 500,
    message: readable || 'Failed to reach Gemini API.',
  };
}

function isTransientError(error) {
  const raw = error instanceof Error ? error.message : String(error || '');
  return raw.includes('503') || raw.includes('UNAVAILABLE') || raw.includes('500') || raw.includes('ECONNRESET');
}

async function withRetry(fn, retries = 1) {
  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < retries && isTransientError(error)) {
        await new Promise((resolve) => setTimeout(resolve, 350));
        continue;
      }
      throw error;
    }
  }
  throw lastError;
}

async function streamChatModel(ai, model, thinkingLevel, contents, systemInstruction, onDelta, abortSignal) {
  const stream = await ai.models.generateContentStream({
    model,
    contents,
    config: {
      systemInstruction,
      thinkingConfig: { thinkingLevel },
      abortSignal,
    },
  });

  let emittedAny = false;
  for await (const chunk of stream) {
    const delta = chunk?.text || '';
    if (delta) {
      emittedAny = true;
      onDelta(delta);
    }
  }
  return emittedAny;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '12mb' }));

  app.get('/api/health', (_req, res) => {
    res.json({
      ok: true,
      geminiConfigured: Boolean(process.env.GEMINI_API_KEY),
    });
  });

  app.post('/api/chat', async (req, res) => {
    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'GEMINI_API_KEY is not configured on the server. Add it in Settings > Secrets.',
      });
      return;
    }

    const { messages, detectedObjects, imageDataUrl, mediaMode } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      res.status(400).json({ error: 'Conversation messages are required.' });
      return;
    }

    const sanitizedHistory = messages
      .filter((m) => m && (m.role === 'user' || m.role === 'model') && typeof m.text === 'string' && m.text.trim().length > 0)
      .slice(-16);

    if (sanitizedHistory.length === 0) {
      res.status(400).json({ error: 'At least one non-empty message is required.' });
      return;
    }

    const detectionSummary = Array.isArray(detectedObjects) && detectedObjects.length > 0
      ? detectedObjects
          .map((obj) => `${obj.label} (${Math.round(Number(obj.confidence) || 0)}%)`)
          .join(', ')
      : 'None currently detected';

    const modeLabel = mediaMode === 'camera'
      ? 'Live camera feed'
      : mediaMode === 'image'
        ? 'Uploaded image'
        : 'No active visual input';

    const systemInstruction = [
      'You are a fast, helpful AI Vision and Knowledge Assistant inside an object detection workspace.',
      'Give clear, direct, concise answers in natural language.',
      `Current visual input mode: ${modeLabel}.`,
      `Objects detected right now: ${detectionSummary}.`,
      'If an image frame is attached, inspect it directly to answer visual questions accurately. For general knowledge questions, answer immediately and clearly.',
    ].join('\n');

    const contents = sanitizedHistory.map((msg, idx) => {
      const isLastUser = idx === sanitizedHistory.length - 1 && msg.role === 'user';
      const parts = [];
      if (isLastUser && imageDataUrl) {
        const parsedImage = parseDataUrl(imageDataUrl);
        if (parsedImage) {
          parts.push({
            inlineData: {
              mimeType: parsedImage.mimeType,
              data: parsedImage.data,
            },
          });
        }
      }
      parts.push({ text: msg.text });
      return {
        role: msg.role,
        parts,
      };
    });

    res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('X-Accel-Buffering', 'no');
    res.flushHeaders?.();

    const writeEvent = (event, data) => {
      res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
    };

    const clientAbort = new AbortController();
    res.on('close', () => {
      if (!res.writableEnded) {
        clientAbort.abort();
      }
    });

    try {
      // Primary ultra-fast low-latency model: gemini-3.1-flash-lite (minimal thinking = instant first token),
      // or gemini-4-hydrogen with ThinkingLevel.LOW if flash-lite is unavailable.
      let emitted = false;
      const primaryController = new AbortController();
      const firstTokenTimer = setTimeout(() => {
        if (!emitted) primaryController.abort();
      }, 6000);

      const onAbort = () => primaryController.abort();
      clientAbort.signal.addEventListener('abort', onAbort);

      const t0 = Date.now();
      let primaryErrMsg = '';
      try {
        emitted = await streamChatModel(
          ai,
          'gemini-3.1-flash-lite',
          ThinkingLevel.MINIMAL,
          contents,
          systemInstruction,
          (delta) => {
            clearTimeout(firstTokenTimer);
            writeEvent('delta', { text: delta });
          },
          primaryController.signal
        );
      } catch (primaryErr) {
        clearTimeout(firstTokenTimer);
        primaryErrMsg = `[flash-lite ${Date.now() - t0}ms: ${primaryErr?.message || primaryErr}]`;
        if (clientAbort.signal.aborted) {
          return;
        }
        // Fallback to gemini-4-hydrogen with ThinkingLevel.LOW
        if (!emitted) {
          const t1 = Date.now();
          try {
            await streamChatModel(
              ai,
              'gemini-4-hydrogen',
              ThinkingLevel.LOW,
              contents,
              systemInstruction,
              (delta) => {
                writeEvent('delta', { text: delta });
              },
              clientAbort.signal
            );
          } catch (fallbackErr) {
            throw new Error(`${primaryErrMsg} [hydrogen ${Date.now() - t1}ms: ${fallbackErr?.message || fallbackErr}]`);
          }
        } else {
          throw primaryErr;
        }
      } finally {
        clearTimeout(firstTokenTimer);
        clientAbort.signal.removeEventListener('abort', onAbort);
      }

      writeEvent('done', { ok: true });
      res.end();
    } catch (error) {
      if (clientAbort.signal.aborted) {
        res.end();
        return;
      }
      const formatted = formatGeminiError(error);
      writeEvent('error', { message: formatted.message });
      res.end();
    }
  });

  app.post('/api/vision/detect', async (req, res) => {
    const ai = createGeminiClient();
    if (!ai) {
      res.status(503).json({
        error: 'GEMINI_API_KEY is not configured on the server.',
      });
      return;
    }

    const { imageDataUrl } = req.body || {};
    const parsedImage = parseDataUrl(imageDataUrl);
    if (!parsedImage) {
      res.status(400).json({ error: 'Valid base64 image data URL is required.' });
      return;
    }

    try {
      const response = await ai.models.generateContent({
        model: 'gemini-4-hydrogen',
        contents: {
          parts: [
            {
              inlineData: {
                mimeType: parsedImage.mimeType,
                data: parsedImage.data,
              },
            },
            {
              text: 'Detect all clearly visible physical objects, people, animals, plants, furniture, and everyday items in this image. Return tight bounding boxes in normalized 0..1000 coordinates [ymin, xmin, ymax, xmax], concise object labels, and calibrated confidence scores between 0 and 1.',
            },
          ],
        },
        config: {
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                label: { type: Type.STRING },
                confidence: { type: Type.NUMBER },
                box_2d: {
                  type: Type.ARRAY,
                  items: { type: Type.NUMBER },
                },
              },
              required: ['label', 'confidence', 'box_2d'],
            },
          },
        },
      });

      const rawText = (response.text || '[]').trim();
      const parsed = JSON.parse(rawText);
      const detections = Array.isArray(parsed)
        ? parsed
            .filter(
              (item) =>
                item &&
                typeof item.label === 'string' &&
                Array.isArray(item.box_2d) &&
                item.box_2d.length === 4
            )
            .map((item) => ({
              label: item.label.trim(),
              score: Math.max(0.01, Math.min(0.99, Number(item.confidence) || 0.85)),
              box_2d: item.box_2d.map((v) => Math.max(0, Math.min(1000, Number(v) || 0))),
            }))
        : [];

      res.json({ detections });
    } catch (error) {
      const formatted = formatGeminiError(error);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  const PROJECT_EXPORT_FILES = [
    '.env.example',
    '.gitignore',
    'index.html',
    'metadata.json',
    'package.json',
    'server.mjs',
    'tsconfig.json',
    'vite.config.ts',
    'src/app.js',
    'src/chat.js',
    'src/detector.js',
    'src/index.css',
    'src/theme.js',
  ];

  const EXPORT_README = [
    '# AI Vision — Live Object Detection & AI Chatbot',
    '',
    '## Quick Start (Self-Hosting)',
    '',
    '1. Install dependencies:',
    '   ```bash',
    '   npm install',
    '   ```',
    '',
    '2. Create a `.env` file in the project root with your own Gemini API key (from https://aistudio.google.com/apikey):',
    '   ```env',
    '   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"',
    '   PORT=3000',
    '   ```',
    '',
    '3. Run in development or production:',
    '   ```bash',
    '   npm run dev',
    '   # or for production:',
    '   npm run build && npm start',
    '   ```',
    '',
    'Note: Object detection (TensorFlow.js COCO-SSD) runs directly in the browser.',
    'The AI Chatbot calls `/api/chat` on `server.mjs`, so host with Node.js (e.g., Render, Railway, Cloud Run, Fly.io, or a VPS) and set `GEMINI_API_KEY` in environment variables.',
    '',
  ].join('\n');

  function crc32(buf) {
    let crc = 0xffffffff;
    for (let i = 0; i < buf.length; i++) {
      crc ^= buf[i];
      for (let j = 0; j < 8; j++) {
        crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function buildZipBuffer(entries) {
    const localParts = [];
    const centralParts = [];
    let offset = 0;

    for (const entry of entries) {
      const nameBuf = Buffer.from(entry.name, 'utf8');
      const rawData = Buffer.isBuffer(entry.data) ? entry.data : Buffer.from(entry.data, 'utf8');
      const compressed = zlib.deflateRawSync(rawData);
      const crc = crc32(rawData);

      const localHeader = Buffer.alloc(30);
      localHeader.writeUInt32LE(0x04034b50, 0);
      localHeader.writeUInt16LE(20, 4);
      localHeader.writeUInt16LE(0x0800, 6);
      localHeader.writeUInt16LE(8, 8);
      localHeader.writeUInt16LE(0, 10);
      localHeader.writeUInt16LE(0, 12);
      localHeader.writeUInt32LE(crc, 14);
      localHeader.writeUInt32LE(compressed.length, 18);
      localHeader.writeUInt32LE(rawData.length, 22);
      localHeader.writeUInt16LE(nameBuf.length, 26);
      localHeader.writeUInt16LE(0, 28);

      localParts.push(localHeader, nameBuf, compressed);

      const centralHeader = Buffer.alloc(46);
      centralHeader.writeUInt32LE(0x02014b50, 0);
      centralHeader.writeUInt16LE(20, 4);
      centralHeader.writeUInt16LE(20, 6);
      centralHeader.writeUInt16LE(0x0800, 8);
      centralHeader.writeUInt16LE(8, 10);
      centralHeader.writeUInt16LE(0, 12);
      centralHeader.writeUInt16LE(0, 14);
      centralHeader.writeUInt32LE(crc, 16);
      centralHeader.writeUInt32LE(compressed.length, 20);
      centralHeader.writeUInt32LE(rawData.length, 24);
      centralHeader.writeUInt16LE(nameBuf.length, 28);
      centralHeader.writeUInt16LE(0, 30);
      centralHeader.writeUInt16LE(0, 32);
      centralHeader.writeUInt16LE(0, 34);
      centralHeader.writeUInt16LE(0, 36);
      centralHeader.writeUInt32LE(0, 38);
      centralHeader.writeUInt32LE(offset, 42);

      centralParts.push(centralHeader, nameBuf);
      offset += localHeader.length + nameBuf.length + compressed.length;
    }

    const centralDir = Buffer.concat(centralParts);
    const eocd = Buffer.alloc(22);
    eocd.writeUInt32LE(0x06054b50, 0);
    eocd.writeUInt16LE(0, 4);
    eocd.writeUInt16LE(0, 6);
    eocd.writeUInt16LE(entries.length, 8);
    eocd.writeUInt16LE(entries.length, 10);
    eocd.writeUInt32LE(centralDir.length, 12);
    eocd.writeUInt32LE(offset, 16);
    eocd.writeUInt16LE(0, 20);

    return Buffer.concat([...localParts, centralDir, eocd]);
  }

  app.get('/api/export-project', (req, res) => {
    const code = typeof req.query.code === 'string' ? req.query.code.trim() : '';
    if (code !== 'BRUHHD2DCODEFILES') {
      res.status(403).json({ error: 'Invalid export code.' });
      return;
    }

    const entries = [{ name: 'ai-vision-project/README.md', data: EXPORT_README }];
    for (const relPath of PROJECT_EXPORT_FILES) {
      const absPath = path.resolve(__dirname, relPath);
      if (fs.existsSync(absPath)) {
        entries.push({
          name: `ai-vision-project/${relPath}`,
          data: fs.readFileSync(absPath),
        });
      }
    }

    const zipBuf = buildZipBuffer(entries);
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', 'attachment; filename="ai-vision-project.zip"');
    res.setHeader('Content-Length', String(zipBuf.length));
    res.send(zipBuf);
  });

  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Aperture Vision server listening on http://0.0.0.0:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
