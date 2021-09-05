def dec(func):
	def wrap():
		print("before func")
		func()
		print("after func")
	return wrap

@dec
def function():
	print("func")

function()