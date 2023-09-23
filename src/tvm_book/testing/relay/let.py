from tvm import relay

x = relay.var("x")
sb = relay.ScopeBuilder()
v1 = sb.let("v1", relay.log(x))
v2 = sb.let("v2", v1 + v1)
sb.ret(v2)
f = relay.Function([x], sb.get())
