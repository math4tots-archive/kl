#import "Cocoa/Cocoa.h"

@interface AppDelegate: NSObject <NSApplicationDelegate> {
  NSWindow* window;
}
@end

@interface View: NSView {
}
@end

static NSBitmapImageRep* makeLayer(double width, double height) {
  return [[NSBitmapImageRep alloc]
    initWithBitmapDataPlanes: NULL
    pixelsWide: (NSInteger) width
    pixelsHigh: (NSInteger) height
    bitsPerSample: 8
    samplesPerPixel: 4
    hasAlpha: YES
    isPlanar: NO
    colorSpaceName: NSDeviceRGBColorSpace
    bytesPerRow: 4 * width
    bitsPerPixel: 32];
}

@implementation AppDelegate
- (AppDelegate*)init {
  if (self = [super init]) {
    window = [[NSWindow alloc]
      initWithContentRect: NSMakeRect(0, 0, 300, 300)
      styleMask:
        NSWindowStyleMaskTitled |
        NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable |
        NSWindowStyleMaskResizable
      backing: NSBackingStoreBuffered
      defer: NO];
    window.contentView = [[View alloc] init];
  }
  return self;
}
- (void)applicationWillFinishLaunching:(NSNotification *)notification {
  window.title = NSProcessInfo.processInfo.processName;
  [window cascadeTopLeftFromPoint: NSMakePoint(20,20)];
  [window makeKeyAndOrderFront: self];
}
@end

@implementation View
- (void)drawRect:(NSRect)dirtyRect {
  // NSBitmapImageRep* rep = makeLayer(300, 300);
  // NSGraphicsContext* gc = [NSGraphicsContext graphicsContextWithBitmapImageRep:rep];
  // [gc saveGraphicsState];
  // [NSColor.blueColor setFill];
  // NSRectFill(NSMakeRect(0, 0, 300, 300));
  // [gc restoreGraphicsState];
  // [rep draw];
  NSImage* image = [[NSImage alloc] initWithSize: NSMakeSize(300, 300)];
  [image lockFocus];
  [NSColor.blueColor setFill];
  NSRectFill(NSMakeRect(0, 0, 300, 300));
  [image unlockFocus];
  [image
    drawAtPoint:NSMakePoint(0, 0)
    fromRect:NSMakeRect(0, 0, 300, 300)
    operation: NSCompositingOperationCopy
    fraction:1.0];
}
@end

int main() {
  @autoreleasepool {
    NSApplication* app = NSApplication.sharedApplication;
    app.ActivationPolicy = NSApplicationActivationPolicyRegular;
    NSMenuItem* item = NSMenuItem.new;
    NSApp.mainMenu = NSMenu.new;
    item.submenu = NSMenu.new;
    [app.mainMenu addItem: item];
    [item.submenu addItem: [[NSMenuItem alloc]
      initWithTitle: [@"Quit "
          stringByAppendingString: NSProcessInfo.processInfo.processName]
      action:@selector(terminate:) keyEquivalent:@"q"]];
    app.delegate = [[AppDelegate alloc] init];
    [NSApp run];
  }
}
