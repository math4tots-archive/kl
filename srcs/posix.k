#(append C_HEADERS 'klc_posix.h')
#(append C_SOURCES 'klc_posix.c')


extern class POSIXInterface {
  String getcwd()

  // Determines what version of POSIX is available
  int GETVERSION()

  // Returns [ino_t, String] pair
  // of directory entries
  List readdir(String path)

  List listdir(String path) {
    final ret = []
    final dirs = this.readdir(path)
    for (int i = 0; i < dirs.size; i++) {
      ret.push(dirs[i][1])
    }
    return ret
  }

  String Repr() {
    return '<POSIX interface>'
  }
}

extern POSIXInterface POSIX
