#include "klc_hacks_gui.h"
#include <windows.h>

#pragma comment(lib, "User32.lib")
#pragma comment(lib, "Gdi32.lib")

static LPCWSTR lpszClassName = L"KlcWindowClassName";

struct KLCNhacksZBguiZBApi {
  KLC_header header;
  HINSTANCE hInstance;
  ATOM windowClassName;
};

static LRESULT CALLBACK WndProc(
    _In_ HWND   hWnd,
    _In_ UINT   uMsg,
    _In_ WPARAM wParam,
    _In_ LPARAM lParam) {
  /* TODO */
  PAINTSTRUCT ps;
  WCHAR greeting[] = L"Hello world!";
  HDC hdc;

  switch (uMsg) {
    case WM_PAINT:
      hdc = BeginPaint(hWnd, &ps);
      TextOutW(hdc, 5, 5, greeting, wcslen(greeting));
      EndPaint(hWnd, &ps);
      break;
    case WM_DESTROY:
      PostQuitMessage(0);
      break;
    default:
      return DefWindowProc(hWnd, uMsg, wParam, lParam);
  }
  return 0;
}

void KLC_deletehacksZBguiZBApi(KLC_header* api, KLC_header** dq) {
}

KLCNhacksZBguiZBApi* KLCNhacksZBguiZBapiZEinit() {
  KLCNhacksZBguiZBApi* api =
    (KLCNhacksZBguiZBApi*) malloc(sizeof(KLCNhacksZBguiZBApi));
  KLC_init_header(&api->header, &KLC_typehacksZBguiZBApi);
  api->hInstance = GetModuleHandle(NULL);

  WNDCLASSEXW wcex;
  wcex.cbSize = sizeof(WNDCLASSEX);
  wcex.style = CS_HREDRAW | CS_VREDRAW;
  wcex.lpfnWndProc = WndProc;
  wcex.cbClsExtra = 0;
  wcex.cbWndExtra = sizeof(LONG_PTR);
  wcex.hInstance = api->hInstance;
  wcex.hIcon = LoadIcon(api->hInstance, IDI_APPLICATION);
  wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
  wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW+1);
  wcex.lpszMenuName = NULL;
  wcex.lpszClassName = lpszClassName;
  wcex.hIconSm = LoadIcon(wcex.hInstance, IDI_APPLICATION);
  api->windowClassName = RegisterClassExW(&wcex);
  if (!api->windowClassName) {
    /* TODO: Return a Try instead */
    KLC_errorf("Call to RegisterClassEx failed");
  }

  return api;
}

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message) {
  MessageBoxW(
    NULL,
    KLC_windows_get_wstr(message),
    L"Alert",
    0);
}

void KLCNhacksZBguiZBApiZFmkwin(KLCNhacksZBguiZBApi* api) {
  MSG msg;
  HWND hWnd = CreateWindowW(
    lpszClassName,
    L"Title",
    WS_OVERLAPPEDWINDOW,
    CW_USEDEFAULT, CW_USEDEFAULT,
    500, 100,
    NULL,
    NULL,
    api->hInstance,
    NULL);

  if (!hWnd) {
    KLC_errorf("Call to CreateWindow failed");
  }

  ShowWindow(hWnd, SW_SHOW);
  UpdateWindow(hWnd);

  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}
