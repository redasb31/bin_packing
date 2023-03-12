
class Bin(object):
    def __init__(self, capacity):
        self.initialcapacity=capacity
        self.capacity = capacity
        self.items = []

    def __str__(self):
        return "Bin: capacity = %d" % (self.capacity)
    
    def print_items(self):
        for item in self.items:
            print(item)
    
    def add_item(self, item):
        self.items.append(item)
        self.capacity -= item.size
    
    def remove_item(self, item):
        self.items.remove(item)
        self.capacity += item.size

class Item(object):
    def __init__(self, size):
        self.size = size

    def __str__(self):
        return "Item: size = %d" % (self.size)