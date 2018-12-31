#if !__has_feature(objc_arc)
#error "Objective-C ARC required!"
#endif

#include "klx/g/lib.osx.h"
#include <stdio.h>
#include <Cocoa/Cocoa.h>


static id KLC_cocoa_make_window();
static int KLC_cocoa_id_to_int(id x);
static KLC_var KLC_cocoa_id_to_var(id x);

@interface KLCWindow : NSWindow
@end

@interface KLCDelegate : NSObject <NSWindowDelegate>
@property (assign) KLC_var implementation;
@end

@interface KLCView : NSView
@property (strong) KLCDelegate* delegate;
@end

static NSMutableArray* KLC_cocoa_refs;

static KLCWindow* KLC_cocoa_make_window(KLCDelegate* delegate) {
  KLCWindow* window = [
    [KLCWindow alloc]
      initWithContentRect:NSMakeRect(200, 200, 200, 200)
      styleMask:NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
      backing:NSBackingStoreBuffered
      defer: NO
  ];
  KLCView* view = [[KLCView alloc] init];
  view.delegate = delegate;
  [window setDelegate:delegate];
  [window setContentView: view];
  [window setTitle:@"Hello, world!"];
  [window orderFrontRegardless];
  return window;
}

static int KLC_cocoa_id_to_int(id x) {
  int ret;
  if (KLC_cocoa_refs == nil) {
    KLC_cocoa_refs = [[NSMutableArray alloc] init];
  }
  ret = KLC_cocoa_refs.count;
  [KLC_cocoa_refs addObject:x];
  return ret;
}

static KLC_var KLC_cocoa_id_to_var(id x) {
  return KLC_cocoa_int_to_var(KLC_cocoa_id_to_int(x));
}

@implementation KLCWindow
@end

@implementation KLCDelegate
- (KLCDelegate*)initWithImplementation:(KLC_var)impl {
  if (self = [super init]) {
    self.implementation = impl;
    KLC_retain_var(impl);
  }
  return self;
}

- (void)dealloc {
  KLC_release_var(self.implementation);
}

- (void)windowWillClose:(NSNotification *)notification {
  [[NSApplication sharedApplication] terminate:self];
}
@end

@implementation KLCView
- (void)drawRect:(NSRect)dirtyRect {
  KLC_var impl = self.delegate.implementation;

  if (KLC_has_method(impl, "draw")) {
    KLC_var out;
    KLC_var args[3];
    KLC_Error* error;
    KLC_Stack* stack = KLC_new_stack();

    args[0] = impl;
    args[1] = KLC_cocoa_drawing_context();
    args[2] = KLC_mklist();

    KLC_list_push(args[2], KLC_var_from_float(dirtyRect.origin.x));
    KLC_list_push(args[2], KLC_var_from_float(dirtyRect.origin.y));
    KLC_list_push(args[2], KLC_var_from_float(dirtyRect.size.width));
    KLC_list_push(args[2], KLC_var_from_float(dirtyRect.size.height));

    error = KLC_call_method(stack, &out, "draw", 3, args);
    KLC_release_var(args[1]);
    KLC_release_var(args[2]);

    if (error) {
      KLC_panic_with_error(error);
    }
    KLC_release_var(out);
    KLC_delete_stack(stack);
  }
}
@end

void KLC_cocoa_sample_func(KLC_var impl) {
  @autoreleasepool {
    KLCWindow* window;
    NSApplication* app;
    KLCDelegate* delegate = [
      [KLCDelegate alloc] initWithImplementation:impl
    ];
    app = [NSApplication sharedApplication];
    [app setActivationPolicy:NSApplicationActivationPolicyRegular];
    window = KLC_cocoa_make_window(delegate);
    [app run];
  }
}

void KLC_cocoa_release(int i) {
}

void KLC_cocoa_fill_rect(double x, double y, double w, double h) {
  [NSBezierPath fillRect:NSMakeRect(x, y, w, h)];
}

extern void KLC_cocoa_set_color(KLC_int r, KLC_int g, KLC_int b) {
  double rf = r / 255.0;
  double gf = g / 255.0;
  double bf = b / 255.0;
  [[NSColor colorWithCalibratedRed:rf green:gf blue:bf alpha:1.0f] set];
}
