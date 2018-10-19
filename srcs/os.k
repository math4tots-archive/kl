#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern class OperatingSystemInterface {
  String GETname()
  bool GETposix()

  bool isfile(String path)
  bool isdir(String path)

  List listdirOrNull(String path)
  String getcwdOrNull()

  List listdir(String path) {
    final ret = this.listdirOrNull(path)
    if (ret is null) {
      panic('Failed to listdir ' + str(path))
    }
    return ret
  }

  String getcwd() {
    final ret = this.getcwdOrNull()
    if (ret is null) {
      panic('Failed to getcwd')
    }
    return ret
  }

  bool chdirOrFalse(String path)
  bool mkdirOrFalse(String path)

  void chdir(String path) {
    if (!this.chdirOrFalse(path)) {
      panic('chdir failed for ' + str(path))
    }
  }

  void mkdir(String path) {
    if (!this.mkdirOrFalse(path)) {
      panic('mkdir failed for ' + path)
    }
  }

  // Path stuff
  String GETsep()
  String joinpath(List parts) {
    return this.sep.join(parts)
  }

  String Repr() {
    return '<OS interface (' + this.name + ')>'
  }
}

extern OperatingSystemInterface OS
