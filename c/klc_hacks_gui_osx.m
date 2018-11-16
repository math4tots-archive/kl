#include "klc_hacks_gui.h"
#import "Cocoa/Cocoa.h"

/* NOTE: You should always create an autoreleasepool */
/* when making calls to any objective-C libraries. */

/* Uses ARC, and needs a new enough clang version such that */
/* we can store object pointers in struct body */

struct KLCNhacksZBguiZBApi {
  KLC_header header;
};

struct KLCNhacksZBguiZBWindow {
  KLC_header header;
  __unsafe_unretained NSWindow* window;
};

static NSMutableSet<NSWindow*>* windowRetainSet = nil;

static NSMutableSet<NSWindow*>* getWindowRetainSet() {
  if (windowRetainSet == nil) {
    windowRetainSet = [[NSMutableSet alloc] init];
  }
  return windowRetainSet;
}

static KLCNhacksZBguiZBWindow* mkwindow(NSWindow* window) {
  KLCNhacksZBguiZBWindow* win =
    (KLCNhacksZBguiZBWindow*) malloc(sizeof(KLCNhacksZBguiZBWindow));
  KLC_init_header(&win->header, &KLC_typehacksZBguiZBWindow);
  @autoreleasepool {
    [getWindowRetainSet() addObject:window];
  }
  return win;
}


void KLC_deletehacksZBguiZBApi(KLC_header* api, KLC_header** dq) {
}

void KLC_deletehacksZBguiZBWindow(KLC_header *robj, KLC_header** dq) {
  KLCNhacksZBguiZBWindow* win = (KLCNhacksZBguiZBWindow*) robj;
  [getWindowRetainSet() removeObject:win->window];
}

KLCNhacksZBguiZBApi* KLCNhacksZBguiZBapiZEinit() {
  KLCNhacksZBguiZBApi* api =
    (KLCNhacksZBguiZBApi*) malloc(sizeof(KLCNhacksZBguiZBApi));
  KLC_init_header(&api->header, &KLC_typehacksZBguiZBApi);
  return api;
}

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message) {
  @autoreleasepool {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:
      [NSString stringWithUTF8String:message->utf8]];
    [alert runModal];
  }
}

void KLCNhacksZBguiZBApiZFmain(KLCNhacksZBguiZBApi* api) {
  @autoreleasepool {
    [NSApp run];
  }
}

KLCNTry* KLCNhacksZBguiZBApiZFwindowZDtry(
    KLCNhacksZBguiZBApi* api,
    KLCNString* title,
    KLC_int width,
    KLC_int height) {
  KLCNTry* ret;
  KLCNhacksZBguiZBWindow* win;
  @autoreleasepool {
    NSRect windowRect = NSMakeRect(100, 100, width, height);
    NSUInteger windowStyle =
      NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskResizable;
    NSWindow* window = [[NSWindow alloc]
      initWithContentRect:windowRect
      styleMask:windowStyle
      backing:NSBackingStoreBuffered
      defer:NO];

    NSTextView* textView = [[NSTextView alloc] initWithFrame:windowRect];

    if (!window) {
      return KLC_failm("NSWindow creation failed");
    }

    [window setContentView:textView];
    [textView
      insertText: @"Hello OSX/Cocoa world!"
      replacementRange: textView.selectedRange];

    win = mkwindow(window);
    ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) win));
    KLC_release((KLC_header*) win);
  }
  return ret;
}

void KLCNhacksZBguiZBWindowZFshow(KLCNhacksZBguiZBWindow* win) {
  [win->window orderFrontRegardless];
}

void KLCNhacksZBguiZBWindowZFupdate(KLCNhacksZBguiZBWindow* win) {
}
