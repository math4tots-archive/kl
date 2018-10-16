#include "klc_cocoa.h"
#import <Foundation/Foundation.h>


@interface KLC_ApplicationDelegate : NSObject<NSApplicationDelegate, NSWindowDelegate> {
  NSWindow * window;
}
@end

@implementation KLC_ApplicationDelegate : NSObject
- (id)init {
  if (self = [super init]) {
  }
  return self
}
@end


extern KLC_typeinfo KLC_typeCOCOAInterface;

KLCNCOCOAInterface* KLCN_initCOCOA() {
  KLCNCOCOAInterface* cocoa =
    (KLCNCOCOAInterface*) malloc(sizeof(KLCNCOCOAInterface));
  KLC_init_header(&cocoa->header, &KLC_typeCOCOAInterface);
  return cocoa;
}

void KLC_deleteCOCOAInterface(KLC_header* robj, KLC_header** dq) {
}
