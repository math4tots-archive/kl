#(append C_HEADERS 'klc_posix.h')
#(append C_SOURCES 'klc_posix.c')


extern class POSIXInterface {
  String getcwd()
}

extern POSIXInterface POSIX

