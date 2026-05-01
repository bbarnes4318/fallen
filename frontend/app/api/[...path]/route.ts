import { NextRequest, NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';

const BACKEND_API_URL = process.env.BACKEND_API_URL || process.env.BACKEND_INTERNAL_URL || 'http://localhost:8000';

async function handleProxy(request: NextRequest, { params }: { params: Promise<{ path: string[] }> }) {
  const resolvedParams = await params;
  const path = resolvedParams.path.join('/');
  const searchParams = request.nextUrl.searchParams.toString();
  const queryString = searchParams ? `?${searchParams}` : '';
  const url = `${BACKEND_API_URL}/${path}${queryString}`;

  const headers = new Headers(request.headers);
  headers.delete('host');

  const requestInit: RequestInit = {
    method: request.method,
    headers,
    redirect: 'manual',
  };

  if (['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method)) {
    const body = await request.arrayBuffer();
    if (body.byteLength > 0) {
      requestInit.body = body;
    }
  }

  try {
    const backendResponse = await fetch(url, requestInit);
    
    const responseHeaders = new Headers(backendResponse.headers);
    responseHeaders.delete('content-encoding');

    return new NextResponse(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error(`API Proxy Error for ${url}:`, error);
    return NextResponse.json({ error: 'Internal Server Error (Proxy)' }, { status: 500 });
  }
}

export const GET = handleProxy;
export const POST = handleProxy;
export const PUT = handleProxy;
export const PATCH = handleProxy;
export const DELETE = handleProxy;
export const OPTIONS = handleProxy;
