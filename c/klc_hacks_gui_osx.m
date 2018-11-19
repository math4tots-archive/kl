#include "klc_hacks_gui.h"
#import "Cocoa/Cocoa.h"

typedef struct KLCNhacksZBguiZBOptions Options;

/* NOTE: ARC is assumed/required here */

/* NOTE: You should always create an autoreleasepool */
/* when making calls to any objective-C libraries. */

@class KLCOBJCWindow;
@class KLCOBJCView;
@class KLCOBJCAppDelegate;

struct KLCNhacksZBguiZBApi {
  KLC_header header;
  Options* opts;
};

@interface KLCOBJCWindow: NSWindow
@end
@implementation KLCOBJCWindow: NSWindow
@end

@interface KLCOBJCView: NSView {
  KLCOBJCAppDelegate* appDelegate;
}
@end
@implementation KLCOBJCView: NSView
@end

@interface KLCOBJCAppDelegate: NSObject <NSApplicationDelegate> {
  KLCOBJCWindow* window;
  Options* opts;
}
@end
@implementation KLCOBJCAppDelegate: NSObject
- (id) initWithContentRect:(NSRect)windowRect options:(Options*)xopts {
  if (self = [super init]) {
    window = [[KLCOBJCWindow alloc]
      initWithContentRect: windowRect
      styleMask: NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
      backing: NSBackingStoreBuffered
      defer: NO];
    opts = xopts;
    KLC_retain((KLC_header*) opts);
  }
  return self;
}
- (void) dealloc {
  KLC_release((KLC_header*) opts);
}
- (void)applicationWillFinishLaunching:(NSNotification *)notification {
  window.title = NSProcessInfo.processInfo.processName;
  [window cascadeTopLeftFromPoint: NSMakePoint(20,20)];
  [window makeKeyAndOrderFront: self];
}
@end

void KLC_deletehacksZBguiZBApi(KLC_header* api, KLC_header** dq) {
}

KLCNTry* KLCNhacksZBguiZBtryApiZEinit() {
  KLCNTry* ret;
  KLCNhacksZBguiZBApi* api =
    (KLCNhacksZBguiZBApi*) malloc(sizeof(KLCNhacksZBguiZBApi));
  KLC_init_header(&api->header, &KLC_typehacksZBguiZBApi);
  api->opts = NULL;
  ret = KLCNTryZEnew(1, KLC_object_to_var((KLC_header*) api));
  KLC_release((KLC_header*) api);
  return ret;
}

void KLCNhacksZBguiZBApiZFalert(KLCNhacksZBguiZBApi* api, KLCNString* message) {
  @autoreleasepool {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:
      [NSString stringWithUTF8String:message->utf8]];
    [alert runModal];
  }
}

void KLCNhacksZBguiZBApiZFstart(KLCNhacksZBguiZBApi* api, Options* opts) {
  @autoreleasepool {
    NSApplication* app = NSApplication.sharedApplication;
    KLCOBJCAppDelegate* appDelegate = [[KLCOBJCAppDelegate alloc]
      initWithContentRect: NSMakeRect(
        KLCNhacksZBguiZBOptionsZFGETx(opts),
        KLCNhacksZBguiZBOptionsZFGETy(opts),
        KLCNhacksZBguiZBOptionsZFGETwidth(opts),
        KLCNhacksZBguiZBOptionsZFGETheight(opts))
      options: opts];
    NSMenuItem* item;

    app.ActivationPolicy = NSApplicationActivationPolicyRegular;
    item = NSMenuItem.new;
    NSApp.mainMenu = NSMenu.new;
    item.submenu = NSMenu.new;
    [app.mainMenu addItem: item];
    [item.submenu addItem: [[NSMenuItem alloc]
      initWithTitle: [@"Quit "
          stringByAppendingString: NSProcessInfo.processInfo.processName]
      action:@selector(terminate:) keyEquivalent:@"q"]];
    app.delegate = appDelegate;
    [NSApp run];
  }
}
