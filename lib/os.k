#(append C_HEADERS 'klc_os.h')
#(append C_SOURCES 'klc_os.c')

extern String name

Try tryChdir(String path)
Try tryMkdir(String path)
Try tryListdir(String path)
Try tryGetcwd()

bool isfile(String path)
bool isdir(String path)

List listdir(String path) {
  return tryListdir(path).orElse(def var[String path](var error) {
    panic('Failed to listdir (' + path + '): ' + str(error))
    return null
  }).(List)
}

String getcwd() {
  return tryGetcwd().orElse(def var(var error) {
    panic('Failed getcwd: ' + str(error))
    return null
  }).(String)
}

void chdir(String path) {
  tryChdir(path).orElse(def var[String path](var error) {
    panic('Failed to chdir (' + path + '): ' + str(error))
    return null
  })
}

void mkdir(String path) {
  tryMkdir(path).orElse(def var[String path](var error) {
    panic('Failed to mkdir (' + path + '): ' + str(error))
    return null
  })
}
