// #(define foo (lambda ()
//   (define x 10)
//   (print 'Hello macro world')
//   x
// ))
// #(foo)
// #(foo)
// #(foo)
// #(foo)
// #(print (foo))

#(assert 1 '1 should be true')

#(define asdf (lambda ()
  (assert 0 '0 is false')
))

#(asdf)

void main() {
}
