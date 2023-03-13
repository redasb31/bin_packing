
class Bin(object):
    def __init__(self, capacity):
        self.initial_capacity=capacity
        self.capacity = capacity
        self.items = []

    def __str__(self):
        return "Bin: capacity = %d" % (self.capacity)
    
    def __hash__(self):
        return hash(self.capacity)

    def print_items(self):
        for item in self.items:
            print(item)
    
    def add_item(self, item):
        self.items.append(item)
        self.capacity -= item.size
    
    def remove_item(self, item):
        self.items.remove(item)
        self.capacity += item.size

    def __eq__(self, other):
        if self.capacity != other.capacity:
            return False
        if len(self.items) != len(other.items):
            return False
        return True

class Item(object):
    def __init__(self, size):
        self.size = size

    def __str__(self):
        return "Item: size = %d" % (self.size)

    def __eq__(self, other):
        return self.size == other.size
    def __lt__(self, other):
        return self.size<other.size