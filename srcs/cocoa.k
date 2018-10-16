#(append APPLE_FRAMEWORKS 'Cocoa')
#(append C_HEADERS 'klc_cocoa.h')
#(cond
  (eq PLATFORM 'darwin')
    (append C_SOURCES 'klc_cocoa.m')
  1
    (append C_SOURCES 'klc_cocoa.c'))


extern class COCOAInterface {
}

extern COCOAInterface COCOA
