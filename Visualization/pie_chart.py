import matplotlib.pyplot as plt

my_data = [583, 254, 1543, 6, 34334, 9332, 2, 217]
my_labels = 'Elektro. Gegengewichtsstapler','Hochhubwagen','Hochhubwagen- KMS', 'Hochregalstapler', 'Niederhubwagen', 'Niederhubwagen-KMS', 'Schlepper', 'Schubmaststapler'

my_explode = (0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0)
pie  = plt.pie(my_data, autopct='%1.1f%%', startangle=15, shadow = True, explode=my_explode)
plt.legend(pie[0], my_labels, loc="lower right", bbox_to_anchor = (0.2,0.4))
plt.title('My Tasks')
plt.axis('equal')
plt.show()