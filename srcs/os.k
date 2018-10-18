#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern class OperatingSystemInterface {
  String GETname()
  bool GETposix()
  List listdir(String path)
  String getcwd()
  bool isfile(String path)
  bool isdir(String path)
  String Repr() {
    return '<OS interface (' + this.name + ')>'
  }
}

extern OperatingSystemInterface OS
