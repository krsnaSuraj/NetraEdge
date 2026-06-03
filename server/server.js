/**
 * NetraEdge Sync Mock — local Express server.
 *
 * Mirrors the AWS API Gateway contract defined in
 * infrastructure/template.yaml. The same RN client
 * (AWSSyncTransport in packages/react-native) talks to this
 * in development and to AWS in production.
 *
 * Endpoints:
 *   POST /sync             — batch upload of pending enrollments
 *   GET  /health           — liveness
 *   GET  /v1/face-sync/pending — list pending server-side records (debug)
 *   GET  /v1/face-sync/stats   — sync stats (debug)
 *
 * Run:
 *   npm install
 *   npm start
 *   # then point RN app at http://10.0.2.2:4000/sync  (Android emulator)
 *   # or http://localhost:4000/sync                   (iOS sim)
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');

const app = express();
app.use(helmet());
app.use(cors());
app.use(express.json({ limit: '4mb' }));
app.use(morgan('dev'));

// In-memory store. Replace with DynamoDB / RDS in production.
const store = {
  enrollments: new Map(),   // userId -> { embedding, metadata, uploadedAt }
  syncStats: { uploads: 0, bytes: 0, lastUploadAt: null, errors: 0 },
};

const API_KEY = process.env.NETRAEDGE_API_KEY || 'demo-key';

function auth(req, res, next) {
  const key = req.header('x-api-key') || req.header('X-Api-Key');
  if (key !== API_KEY) {
    return res.status(401).json({ error: 'invalid_api_key' });
  }
  next();
}

app.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    service: 'netraedge-sync-mock',
    version: '0.1.0',
    uptime_seconds: process.uptime(),
  });
});

/**
 * POST /sync
 * Body: { items: [{ userId, embedding: number[128], metadata }] }
 * Returns: { accepted: number, rejected: number, serverIds: string[] }
 */
app.post('/sync', auth, (req, res) => {
  const items = (req.body && req.body.items) || [];
  if (!Array.isArray(items)) {
    store.syncStats.errors += 1;
    return res.status(400).json({ error: 'items_must_be_array' });
  }
  const accepted = [];
  const rejected = [];
  for (const item of items) {
    const { userId, embedding, metadata } = item || {};
    if (typeof userId !== 'string' || userId.length === 0) {
      rejected.push({ userId, reason: 'missing_userId' });
      continue;
    }
    if (!Array.isArray(embedding) || embedding.length !== 128) {
      rejected.push({ userId, reason: 'embedding_must_be_128_dim' });
      continue;
    }
    const serverId = `srv_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    store.enrollments.set(userId, {
      userId,
      embedding,
      metadata: metadata || {},
      serverId,
      uploadedAt: new Date().toISOString(),
    });
    store.syncStats.uploads += 1;
    store.syncStats.bytes += 4 * embedding.length;
    store.syncStats.lastUploadAt = new Date().toISOString();
    accepted.push({ userId, serverId });
  }
  res.json({
    accepted: accepted.length,
    rejected: rejected.length,
    items: accepted,
    rejected,
  });
});

app.get('/v1/face-sync/pending', auth, (_req, res) => {
  res.json({
    pending: Array.from(store.enrollments.values()).map(({ userId, serverId, uploadedAt }) => ({
      userId, serverId, uploadedAt,
    })),
  });
});

app.get('/v1/face-sync/stats', auth, (_req, res) => {
  res.json({
    ...store.syncStats,
    enrollmentCount: store.enrollments.size,
  });
});

app.get('/v1/face-sync/:userId', auth, (req, res) => {
  const e = store.enrollments.get(req.params.userId);
  if (!e) return res.status(404).json({ error: 'not_found' });
  res.json(e);
});

const PORT = parseInt(process.env.PORT || '4000', 10);
app.listen(PORT, '0.0.0.0', () => {
  // eslint-disable-next-line no-console
  console.log(`[netraedge-sync-mock] listening on http://0.0.0.0:${PORT}`);
  console.log(`[netraedge-sync-mock] API key: ${API_KEY === 'demo-key' ? '(default demo-key)' : '(custom)'}`);
});
