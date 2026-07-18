const CACHE_NAME = 'aegis-scada-cache-v3'; // Incremented cache version
const ASSETS_TO_CACHE = [
  '/manifest.json',
  '/favicon.ico',
];

// Install Service Worker and cache core static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[Service Worker] Caching static shell assets');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
  self.skipWaiting();
});

// Activate Service Worker and IMMEDIATELY destroy old version caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keyList) => {
      return Promise.all(
        keyList.map((key) => {
          // Force deletion of all older cache names
          if (key !== CACHE_NAME) {
            console.log('[Service Worker] Obliterating old cache:', key);
            return caches.delete(key);
          }
        })
      );
    }).then(() => {
      // Force immediate control over current client tabs
      return self.clients.claim();
    })
  );
});

// Fetch events: Network-first with cache fallback strategy for index
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET' || event.request.url.includes('/api/')) {
    return;
  }

  // Network-first strategy for the root index page to ensure updates show up
  const isIndex = new URL(event.request.url).pathname === '/';
  
  if (isIndex) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          return response;
        })
        .catch(() => {
          // If offline, serve from cache if available, else network error
          return caches.match(event.request);
        })
    );
    return;
  }

  // Cache-first for other static assets
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      if (cachedResponse) {
        return cachedResponse;
      }
      return fetch(event.request).then((networkResponse) => {
        if (!networkResponse || networkResponse.status !== 200) {
          return networkResponse;
        }
        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then((cache) => {
          cache.put(event.request, responseToCache);
        });
        return networkResponse;
      });
    })
  );
});
