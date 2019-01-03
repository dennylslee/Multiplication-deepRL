a = 20

b = a // 10
c = a % (b*10)

print(b)
print(c)

# --------------------------------------- extracting each digit-----------------------
product = 1459
product_digit1000 = product // 1000
product_digit100  = (product - (product_digit1000 * 1000)) // 100
product_digit10   = (product - (product_digit1000 * 1000) - (product_digit100 * 100)) // 10
product_digit1    = (product - (product_digit1000 * 1000) - (product_digit100 * 100) - (product_digit10 * 10)) // 1
print(product_digit1000)
print(product_digit100)
print(product_digit10)
print(product_digit1)

print('SAME AS ABOVE')
a = [0,0,0,0]
for i in [3,2,1,0]:
	sum = 0
	for index, k in enumerate(a[0:3-i]): sum += (k * (10**(3-index)))
	a[3-i] = (product - sum) // 10**i
	print(sum)
print('iterate method', a)

# -------------------------  creating list ------------------------------
j = [1, 2, 3 ]
u = [4,5,6] + j
print(u)

#--------------------------- creating list of list -----------------------
digit_range = list(range(10))
y = []
for i in range(4): y.append(digit_range)
print(y)

# -----------------------------

import numpy as np
alist = [4,5,6]
y = np.array([[1,2,3] + alist])
print(y)

# ----------------------------

print('PRODUCT COUNT SECTION')
product = []
a = [1,2,3,4]
b = [1,2,3,4]
for i in a:
	for j in b:
		product.append(i * j)

for i in range(a[-1] * b[-1]):
	print('Count of product', i+1, 'is', product.count(i+1))

