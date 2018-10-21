#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern class Interface {
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

extern Interface OS

bool isdir(String path) {
  return OS.isdir(path)
}

bool isfile(String path) {
  return OS.isfile(path)
}

List listdir(String path) {
  return OS.listdir(path)
}

String name = OS.name
bool posix = OS.posix

String getcwd() {
  return OS.getcwd()
}

String joinpath(List parts) {
  return OS.sep.join(parts)
}

void mkdir(String path) {
  OS.mkdir(path)
}

void chdir(String path) {
  OS.chdir(path)
}
