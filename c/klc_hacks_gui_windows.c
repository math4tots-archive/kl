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

struct KLCNhacksZBguiZBWindow {
  KLC_header header;
  KLCNhacksZBguiZBApi* api;
  HWND hWnd;
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

static KLCNhacksZBguiZBWindow* mkwindow(KLCNhacksZBguiZBApi* api, HWND hWnd) {
  KLCNhacksZBguiZBWindow* win =
    (KLCNhacksZBguiZBWindow*) malloc(sizeof(KLCNhacksZBguiZBWindow));
  KLC_init_header(&win->header, &KLC_typehacksZBguiZBWindow);
  KLC_retain(&api->header);
  win->api = api;
  win->hWnd = hWnd;
  return win;
}

void KLC_deletehacksZBguiZBApi(KLC_header* api, KLC_header** dq) {
}

void KLC_deletehacksZBguiZBWindow(KLC_header* robj, KLC_header** dq) {
  KLCNhacksZBguiZBWindow* win = (KLCNhacksZBguiZBWindow*) robj;
  KLC_partial_release(&win->api->header, dq);
  /*
   TODO: Determine if DestroyWindow should be called here
  */
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


void KLCNhacksZBguiZBApiZFmain(KLCNhacksZBguiZBApi* api) {
  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  exit((int) msg.wParam);
}

KLCNTry* KLCNhacksZBguiZBApiZFwindowZDtry(
    KLCNhacksZBguiZBApi* api,
    KLCNString* title,
    KLC_int width,
    KLC_int height) {
  KLCNhacksZBguiZBWindow* win;
  KLCNTry* ret;
  HWND hWnd = CreateWindowW(
      lpszClassName,
      KLC_windows_get_wstr(title),
      WS_OVERLAPPEDWINDOW,
      CW_USEDEFAULT, CW_USEDEFAULT,
      width, height,
      NULL,
      NULL,
      api->hInstance,
      NULL);

  if (!hWnd) {
    /* TODO: Return Try instead */
    return KLC_failm("Call to CreateWindow failed");
  }

  win = mkwindow(api, hWnd);
  SetWindowLongPtrW(hWnd, 0, (LONG_PTR) win);
  if ((KLCNhacksZBguiZBWindow*) GetWindowLongPtrW(hWnd, 0) != win) {
    KLC_errorf("FUBAR: GetWindowLongPtrW doesn't work as expected");
  }
  ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) win));
  KLC_release((KLC_header*) win);
  return ret;
}

void KLCNhacksZBguiZBWindowZFshow(KLCNhacksZBguiZBWindow* win) {
  ShowWindow(win->hWnd, SW_SHOW);
}

void KLCNhacksZBguiZBWindowZFupdate(KLCNhacksZBguiZBWindow* win) {
  UpdateWindow(win->hWnd);
}
