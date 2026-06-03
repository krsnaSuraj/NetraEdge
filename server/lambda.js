/**
 * AWS Lambda handler — wraps the Express app for SAM.
 *
 * The same server.js can run locally (express) or in Lambda
 * via aws-serverless-express. For SAM we use a simpler shape:
 * route-by-path in a single handler.
 *
 * The full Express app is wrapped with `serverless-http` if you
 * prefer. This file uses the SAM-idiomatic direct handler.
 */

const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
const { DynamoDBDocumentClient, PutCommand, ScanCommand } = require('@aws-sdk/lib-dynamodb');

const ddb = DynamoDBDocumentClient.from(new DynamoDBClient({}));
const TABLE = process.env.TABLE_NAME;

const HEADERS = {
  'content-type': 'application/json',
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'content-type,x-api-key',
  'access-control-allow-methods': 'GET,POST,OPTIONS',
};

function authOk(event) {
  const key = (event.headers || {})['x-api-key'] || (event.headers || {})['X-Api-Key'];
  return key && key === process.env.NETRAEDGE_API_KEY;
}

function respond(status, body) {
  return { statusCode: status, headers: HEADERS, body: JSON.stringify(body) };
}

exports.lambdaHandler = async (event) => {
  // CORS preflight
  if (event.httpMethod === 'OPTIONS') {
    return respond(200, { ok: true });
  }

  const path = event.path || event.rawPath || '/';
  const method = event.httpMethod || event.requestContext?.http?.method;

  if (path === '/health' && method === 'GET') {
    return respond(200, { status: 'ok', service: 'netraedge-sync', version: '0.1.0' });
  }

  if (!authOk(event)) {
    return respond(401, { error: 'invalid_api_key' });
  }

  if (path === '/sync' && method === 'POST') {
    let body;
    try { body = JSON.parse(event.body || '{}'); } catch { return respond(400, { error: 'bad_json' }); }
    const items = Array.isArray(body.items) ? body.items : [];
    const accepted = [];
    const rejected = [];
    for (const it of items) {
      const { userId, embedding, metadata } = it || {};
      if (typeof userId !== 'string' || userId.length === 0) {
        rejected.push({ userId, reason: 'missing_userId' });
        continue;
      }
      if (!Array.isArray(embedding) || embedding.length !== 512) {
        rejected.push({ userId, reason: 'embedding_must_be_512_dim' });
        continue;
      }
      const uploadedAt = new Date().toISOString();
      try {
        await ddb.send(new PutCommand({
          TableName: TABLE,
          Item: { userId, uploadedAt, embedding, metadata: metadata || {} },
        }));
        accepted.push({ userId, uploadedAt });
      } catch (e) {
        rejected.push({ userId, reason: 'ddb_error', detail: String(e) });
      }
    }
    return respond(200, { accepted: accepted.length, rejected: rejected.length, items: accepted, rejected });
  }

  if (path === '/v1/face-sync/stats' && method === 'GET') {
    const r = await ddb.send(new ScanCommand({
      TableName: TABLE,
      Select: 'COUNT',
    }));
    return respond(200, { enrollmentCount: r.Count });
  }

  return respond(404, { error: 'not_found' });
};
