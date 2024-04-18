# Note
All the matrials in this page are generated using the following custom GPT:

🚀 **[Software Interview Mentor](https://chat.openai.com/g/g-n76b8bWJo-software-interview-mentor)** - Learn about different concepts! 🤖 (Requires ChatGPT Plus)

# LeetCode Cheatsheet
🚀 **[LeetCode Cheatsheet](https://leetcode.com/explore/interview/card/cheatsheets/720/resources/4723/)** - Great resource for templates of how to write code for each algorithm pattern. Super helpful to review!

# Sliding Window

The sliding window technique is a method used to solve problems that involve arrays or lists, especially when you're asked to find a subarray that satisfies certain conditions. This technique is particularly useful for problems where you need to consider contiguous elements together. The key idea is to maintain a 'window' that slides over the data to examine different subsets of it.

### Basic Concept:
- **Window**: A continuous portion of the array/list.
- **Sliding**: Moving the window's start and end points one element at a time.

### Usage:
1. **Fixed-Size Window**: The window size remains constant as it slides. For example, finding the maximum sum of any consecutive `k` elements.
2. **Variable-Size Window**: The window size changes based on certain conditions. For example, finding the smallest subarray with a sum greater than a given value.

### Python Example - Fixed Size Window:
Let's look at an example where you need to find the maximum sum of any consecutive `k` elements in an array.

```python
def max_sum_subarray(arr, k):
    # Initialize max_sum to 0. This will store the maximum sum found.
    max_sum = 0

    # Calculate the sum of the first 'k' elements in the array.
    # This is our initial window sum.
    window_sum = sum(arr[:k])

    # Loop through the array, but only until len(arr) - k.
    # This is because we are considering 'k' elements at a time,
    # and we stop when we reach the last 'k' elements.
    for i in range(len(arr) - k):
        # Slide the window forward by one element:
        # Subtract the element going out of the window (arr[i])
        # and add the new element entering into the window (arr[i + k]).
        window_sum = window_sum - arr[i] + arr[i + k]

        # Update max_sum if the current window_sum is greater than the previously recorded max_sum.
        max_sum = max(max_sum, window_sum)

    # Return the maximum sum found.
    return max_sum

# Example usage
arr = [1, 4, 2, 10, 23, 3, 1, 0, 20]  # Input array
k = 4  # Number of consecutive elements to consider
print(max_sum_subarray(arr, k))  # Output will be the maximum sum of 4 consecutive elements
```

### Python Example - Variable Size Window:
Now, let's look at a variable-size window problem, like finding the smallest subarray with a sum greater than a given value.

```python
def smallest_subarray_with_given_sum(arr, s):
    # Initialize min_length with infinity. This variable will hold the length of the smallest subarray.
    min_length = float('inf')

    # Initialize window_sum to 0. It will store the sum of elements in the current window.
    window_sum = 0

    # Initialize window_start to 0. It marks the start of the sliding window.
    window_start = 0

    # Iterate over the array using window_end as the end of the sliding window.
    for window_end in range(len(arr)):
        # Add the current element to the window_sum.
        window_sum += arr[window_end]

        # Shrink the window from the start if the window_sum is greater than or equal to s.
        while window_sum >= s:
            # Update min_length with the smaller length between the previous min_length and current window size.
            min_length = min(min_length, window_end - window_start + 1)

            # Subtract the element at window_start from window_sum and move window_start forward.
            window_sum -= arr[window_start]
            window_start += 1

    # Return min_length if a subarray was found; otherwise, return 0.
    # Checking against float('inf') is necessary to handle the case where no such subarray is found.
    return min_length if min_length != float('inf') else 0

# Example usage
arr = [2, 1, 5, 2, 3, 2]  # Input array
s = 7  # Target sum
print(smallest_subarray_with_given_sum(arr, s))  # Output will be the length of the smallest subarray with sum >= s
```

### Real-World Application:
In large-scale systems, the sliding window technique is often used in areas like network data analysis or real-time analytics, where it's essential to analyze a subset of data in a moving time frame. For example, monitoring the maximum traffic load on a server in any given 10-minute window can help in resource allocation and predicting potential overload scenarios.

## Prefix Sum

The Prefix Sum pattern is a powerful technique in algorithms and data structures, particularly useful in solving problems involving arrays or lists. It's about creating an auxiliary array, the prefix sum array, which stores the sum of elements from the start to each index of the original array. This technique simplifies solving problems related to range sum queries and subarray sums.

### Key Concept:
- **Prefix Sum Array:** Given an array `arr`, its prefix sum array `prefixSum` is defined such that `prefixSum[i]` is the sum of all elements `arr[0]`, `arr[1]`, ..., `arr[i]`.

### Advantages:
1. **Efficient Range Queries:** Once the prefix sum array is built, you can quickly find the sum of elements in a range `[i, j]` by simply calculating `prefixSum[j] - prefixSum[i-1]`.
2. **Preprocessing Time-Saver:** Building the prefix sum array takes O(N) time, but once built, range sum queries are O(1).
3. **Versatility:** Useful in various scenarios like calculating cumulative frequency, image processing, and more.

### Real-World Example:
Consider a large-scale system like a finance tracking app. You need to quickly calculate the total expenditure over different time ranges. By using a prefix sum array of daily expenses, you can rapidly compute the sum over any date range, enhancing the performance of the app.

### Python Example:
Let's create a prefix sum array and use it to find a range sum.

```python
def create_prefix_sum(arr):
    # Initialize the prefix sum array with the first element of arr
    prefix_sum = [arr[0]] 

    # Compute the prefix sum array
    for i in range(1, len(arr)):
        prefix_sum.append(prefix_sum[i-1] + arr[i])

    return prefix_sum

def range_sum_query(prefix_sum, start, end):
    # Handle the case when start is 0
    if start == 0:
        return prefix_sum[end]
    return prefix_sum[end] - prefix_sum[start - 1]

# Example usage
arr = [3, 1, 4, 1, 5, 9, 2, 6]
prefix_sum = create_prefix_sum(arr)

# Get the sum of elements from index 2 to 5
print(range_sum_query(prefix_sum, 2, 5))  # Output: 19
```

### LeetCode Style Question:
**Problem - "Subarray Sum Equals K" (LeetCode 560):** Given an array of integers `nums` and an integer `k`, return the total number of continuous subarrays whose sum equals to `k`.

**Solution:**
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    sum_freq = {0: 1}

    for num in nums:
        prefix_sum += num
        # Check if there is a prefix sum that, when subtracted from the current prefix sum, equals k
        if prefix_sum - k in sum_freq:
            count += sum_freq[prefix_sum - k]

        # Update the frequency of the current prefix sum
        sum_freq[prefix_sum] = sum_freq.get(prefix_sum, 0) + 1

    return count

# Example usage
print(subarraySum([1, 1, 1], 2))  # Output: 2
```
In this problem, we use a hash map (`sum_freq`) to store the frequency of prefix sums. As we iterate through the array, we check if `prefix_sum - k` is in our map. If it is, it means there are one or more subarrays ending at the current index which sum up to `k`. This approach is efficient and showcases the utility of the prefix sum pattern in solving complex problems.

## Hash Map / Set
The Hash Map/Set pattern in interviews typically revolves around leveraging hash tables to efficiently store, access, and manipulate data. Hash tables, implemented in Python as dictionaries (hash maps) and sets (hash sets), provide average time complexity of O(1) for insert, delete, and lookup operations, making them incredibly efficient for certain types of problems.

### Key Characteristics of Hash Map/Set Pattern:

1. **Efficiency**: The direct access nature of hash maps/sets allows for faster data retrieval compared to linear structures like arrays or linked lists.
2. **Uniqueness**: Sets naturally enforce uniqueness of elements, making them ideal for solving problems involving deduplication or presence checks.
3. **Key-Value Storage**: Hash maps store data in key-value pairs, allowing for efficient data association and retrieval. This is useful for counting frequencies, mapping relationships, etc.
4. **Ordering**: Standard hash maps/sets in Python (as of Python 3.7) maintain insertion order, but it's crucial to remember that the primary feature of hash tables is not ordering but fast access.

### Real-World Example:

Consider a web service that tracks the number of views for various videos. A hash map could efficiently map video IDs to view counts, allowing the service to quickly update or retrieve views for any video. This is critical in large-scale systems where performance and scalability are paramount.

### Common Interview Problems and Solutions:

#### Problem 1: Find the First Unique Character in a String

Given a string, find the first non-repeating character in it and return its index. If it doesn't exist, return -1.

**Solution**:

- Use a hash map to count the frequency of each character.
- Iterate through the string to find the first character with a frequency of 1.

```python
def firstUniqChar(s: str) -> int:
    # Build a hash map to store character frequencies
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the first unique character
    for index, char in enumerate(s):
        if char_count[char] == 1:
            return index

    return -1
```

#### Problem 2: Contains Duplicate

Given an array of integers, find if the array contains any duplicates.

**Solution**:

- Use a set to track seen numbers.
- If a number is already in the set, a duplicate exists.

```python
def containsDuplicate(nums: List[int]) -> bool:
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False
```

### Conclusion

The Hash Map/Set pattern is powerful for problems involving data access and manipulation due to its efficiency and flexibility. By understanding and applying this pattern, you can solve a wide range of problems more effectively in your interviews. Remember to analyze the problem's requirements carefully to determine when using a hash map or set is the most appropriate solution.

# Stack
The Stack interview pattern involves using a stack data structure to solve problems that require you to process elements in a Last-In-First-Out (LIFO) manner. This pattern is particularly useful in scenarios where you need to keep track of previously seen elements in a way that the last element you encounter is the first one you need to retrieve for processing.

### Key Concepts

1. **LIFO Principle**: The last element added to the stack is the first one to be removed.
2. **Operations**: The primary operations involved with stacks are:
   - `push()`: Add an element to the top of the stack.
   - `pop()`: Remove the top element from the stack.
   - `peek()` or `top()`: View the top element without removing it.
   - `isEmpty()`: Check if the stack is empty.

### Use Cases

- **Parentheses Matching**: Checking for balanced parentheses in an expression.
- **Undo Mechanism**: In text editors, browsers, etc., where the last action can be undone.
- **Function Call Management**: Managing function calls in programming languages, where the call stack is a stack.
- **Histogram Problems**: Calculating maximum area under histograms.
- **String Manipulations**: Reversing strings or checking for palindromes.

### Python Example: Checking for Balanced Parentheses

Let's explore a common interview question solved using the stack pattern: checking if an expression has balanced parentheses.

```python
def isBalanced(expression):
    # Stack to keep track of opening brackets
    stack = []
    
    # Mapping of closing to opening brackets
    mapping = {')': '(', '}': '{', ']': '['}
    
    for char in expression:
        if char in mapping:
            # Pop the top element if the stack isn't empty, else assign a dummy value
            top_element = stack.pop() if stack else '#'
            
            # Check if the popped element matches the mapping
            if mapping[char] != top_element:
                return False
        else:
            # Push the opening bracket onto the stack
            stack.append(char)
    
    # The expression is balanced if the stack is empty
    return not stack

# Example usage
expression = "{[()()]}"
print(isBalanced(expression))  # Output: True
```

### Real-world Example

Imagine implementing a feature for a text editor that allows users to undo their last set of actions. A stack can be used to store actions as they occur. When the user triggers the undo function, the most recent action is popped from the stack and reversed. This LIFO approach ensures that actions are undone in the reverse order they were made, which is a common expectation in user interfaces.

### Solving a LeetCode Problem: Largest Rectangle in Histogram

One of the more challenging problems that can be solved using the stack pattern is finding the largest rectangle in a histogram. This involves processing bars in a histogram to find the largest rectangle that can be formed within the bounds of the histogram. The problem definition: Given an array of integers (heights) representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram. The challenging part is that different bars from the histogram can be combined to represent a larger rectangle as visualized in [this Leetcode problem](https://leetcode.com/problems/largest-rectangle-in-histogram/description/). Good explanation by [NeetCode](https://www.youtube.com/watch?v=zx5Sw9130L0).

```python
def largestRectangleArea(heights):
    stack = []  # Create a stack to keep indices of the bars
    max_area = 0  # Initialize max area as zero
    
    # Iterate through all bars of the histogram
    for i, h in enumerate(heights):
        start = i
        while stack and stack[-1][1] > h:
            index, height = stack.pop()
            max_area = max(max_area, height * (i - index))
            start = index
        stack.append((start, h))
    
    # Compute area for the remaining bars in stack
    for i, h in stack:
        max_area = max(max_area, h * (len(heights) - i))
    
    return max_area

# Example usage
heights = [2,1,5,6,2,3]
print(largestRectangleArea(heights))  # Output: 10
```

In this code, we maintain a stack to keep track of bars. When we see a bar that is lower than the bar at the top of the stack, we start calculating the area with the bar at the top as the smallest bar. We do this because the current bar stops the previous bars from extending further. This solution efficiently processes each bar and determines the area of the largest rectangle that can be formed.

### LIFO Stack with List

Implementing a LIFO stack with a list is straightforward since lists naturally support append and pop operations at the end, which are efficient and align with the LIFO principle.

```python
# LIFO Stack Implementation
stack = []

# Push items onto the stack
stack.append('A')
stack.append('B')
stack.append('C')

# Pop an item off the stack
last_in = stack.pop()
print("Popped Item:", last_in)  # C

# The stack now contains: ['A', 'B']
```

### FIFO Queue with List

For a FIFO queue, you can still use a list, but you should be aware of the performance implications. Using `append()` to enqueue and `pop(0)` to dequeue will work, but `pop(0)` has a linear time complexity (O(n)) because it requires shifting all other elements by one.

```python
# FIFO Queue Implementation (Not Recommended for High Performance Needs)
queue = []

# Enqueue items
queue.append('A')
queue.append('B')
queue.append('C')

# Dequeue an item
first_in = queue.pop(0)
print("Dequeued Item:", first_in)  # A

# The queue now contains: ['B', 'C']
```

### Recommended Approach for FIFO in Interviews

For interviews, it's essential to discuss the efficiency of your data structure choices. If asked to implement a FIFO queue, it’s better to mention or use collections.deque, which is designed to have fast appends and pops from both ends.

```python
from collections import deque

# FIFO Queue Implementation using deque
queue = deque()

# Enqueue items
queue.append('A')
queue.append('B')
queue.append('C')

# Dequeue an item
first_in = queue.popleft()
print("Dequeued Item:", first_in)  # A

# The queue now contains: deque(['B', 'C'])
```

### Summary for Interviews

- For LIFO stack operations, using a list is perfectly fine and recommended due to its simplicity and efficiency for stack-related operations.
- For FIFO queue operations, prefer using `collections.deque` to avoid performance issues associated with list operations that affect the beginning of the list. Mentioning the efficiency concern shows your understanding of underlying data structures and their performance characteristics.

Explaining your choice of data structure and being aware of its performance implications can positively impact your interview, demonstrating both your coding skills and your understanding of data structures.

## Trie
The Trie algorithm pattern, often referred to as a prefix tree, is a specialized tree used to handle a dynamic set of strings where keys are usually strings. Unlike binary search trees, where the position of a node is determined by comparing the less than or greater than relationship to the parent node, in a Trie, the position of a node is determined by the characters in the string it represents. This makes Tries an incredibly efficient data structure for tasks such as autocomplete, spell checking, IP routing, and other applications where prefix matching is important.

### Structure of a Trie

A Trie is a rooted tree with nodes that contain a set of children per node, each representing one character of the alphabet. Here's a basic structure of a Trie node:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
```

- `children`: A dictionary mapping characters to the next TrieNode.
- `is_end_of_word`: A boolean indicating whether this node represents the end of a word in the Trie.

### Basic Operations

#### Insertion

To insert a word into a Trie, start from the root and traverse the Trie following the characters of the word. If a character is not present, create a new node in the corresponding child position. Mark the end node as the end of a word.

```python
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
```

#### Search

To search for a word, traverse the Trie following the characters of the word. If at any step the character is not found, return False. If all characters are found and the last node is marked as the end of a word, return True.

```python
def search(self, word):
    node = self.root
    for char in word:
        if char not in node.children:
            return False
        node = node.children[char]
    return node.is_end_of_word
```

#### Prefix Search

This operation checks whether there is any word in the Trie that starts with the given prefix.

```python
def startsWith(self, prefix):
    node = self.root
    for char in prefix:
        if char not in node.children:
            return False
        node = node.children[char]
    return True
```

### Real-world Example

Consider an autocomplete system, like the ones used in search engines or messaging apps. A Trie can efficiently store a large dictionary of words and quickly retrieve all words that share a common prefix, which is essential for suggesting completions as the user types.

### Example Problem: Implement an Autocomplete System

Let's design a basic autocomplete system using a Trie. For simplicity, we'll focus on inserting words and finding completions for a given prefix.

```python
class AutocompleteSystem:
    def __init__(self, words):
        self.trie = Trie()
        for word in words:
            self.trie.insert(word)
    
    def autocomplete(self, prefix):
        completions = []
        node = self.trie.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        self.dfs(node, prefix, completions)
        return completions
    
    def dfs(self, node, prefix, completions):
        if node.is_end_of_word:
            completions.append(prefix)
        for char, child_node in node.children.items():
            self.dfs(child_node, prefix + char, completions)
```

In this example, `AutocompleteSystem` initializes a Trie with a list of words. The `autocomplete` function finds all words in the Trie that start with a given prefix, using depth-first search to traverse and collect completions.

Tries are a powerful tool for working with strings and can significantly improve the performance and efficiency of your code in scenarios involving prefix matching and word retrieval.

### Example Problem: Equal Row and Column Pairs using Trie
LeetCode problem 2352, "Equal Row and Column Pairs," asks for finding pairs of rows and columns in a square matrix that are identical. At first glance, using a Trie for this problem might not seem intuitive since Tries are typically used for string manipulations or prefix-related queries. However, with a creative approach, we can adapt the Trie data structure to solve this problem efficiently by treating each row and column as a string of numbers.

#### Problem Statement

Given an `n x n` integer matrix `grid`, return the number of pairs `(r, c)` where row `r` and column `c` are identical.

#### Approach

To solve this problem, we'll insert each row of the matrix into a Trie, treating each row as a "word" where each "character" is an element of the row. After inserting all rows, we'll traverse each column of the matrix, checking if the column exists in the Trie as if we were searching for a word.

Here's how we can implement this approach:

1. **Trie Node Structure**: Each Trie node will hold a dictionary mapping the next digit to the next Trie node, and a count to track how many times a "word" (in this case, a row) ends at this node.

2. **Insert Rows**: For each row in the grid, insert it into the Trie.

3. **Search Columns**: For each column, traverse the Trie. If we can successfully traverse the Trie using the column's elements as the path and find nodes that represent the end of rows, we increment our pairs count based on the count stored in the final node of that path.

#### Python Implementation

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.endCount = 0  # Tracks how many rows end at this node

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, row):
        node = self.root
        for num in row:
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]
        node.endCount += 1  # Mark the end of a row and count it

def equalPairs(grid):
    n = len(grid)
    trie = Trie()
    
    # Insert all rows into the Trie
    for row in grid:
        trie.insert(row)
    
    pairCount = 0
    
    # Check each column against the Trie
    for c in range(n):
        node = trie.root
        for r in range(n):
            if grid[r][c] in node.children:
                node = node.children[grid[r][c]]
            else:
                break  # This column does not match any row
        else:  # If we didn't break, this column matches a row
            pairCount += node.endCount
    
    return pairCount
```

#### Explanation

- **Inserting Rows**: We insert each row into the Trie, treating each element of the row as a part of a path in the Trie. The `endCount` at the last node of each path is incremented to indicate the end of a row and how many times it appears.
  
- **Searching for Columns**: For each column, we attempt to follow a path in the Trie corresponding to the column's elements. If we reach the end of the path (`else` clause of the loop), it means the column matches one or more rows, and we add the `endCount` of the final node to `pairCount`.

This solution leverages the Trie data structure to efficiently compare rows and columns, exploiting the fact that both rows and columns can be treated as sequences of numbers, similar to strings in traditional Trie use cases.

# Queue
Queues are a fundamental data structure that operates on a First In, First Out (FIFO) principle, meaning the first element added to the queue will be the first one to be removed. This characteristic makes queues incredibly useful for managing tasks in sequential order, simulating real-world scenarios like customer service lines, and handling data in streams.

### Real-World Example: Web Server Request Handling

In large-scale systems, like web servers, queues play a critical role in managing incoming requests. When a server receives more requests than it can process simultaneously, it places the excess requests in a queue. This ensures that each request is handled in the order it was received, preventing server overload and maintaining fair access for users.

### Python Implementation

In Python, queues can be implemented using the `queue` module for thread-safe operations, or simply with a list, although the latter is not recommended for production due to performance issues when the list grows.

```python
from queue import Queue

# Initialize a queue
q = Queue()

# Add elements
q.put('A')
q.put('B')

# Remove and return an element
first_element = q.get()

print(first_element)  # Output: 'A'
```

### Queue in Leetcode Problems

Queues are especially useful in solving problems related to graph traversal (like BFS), caching strategies (like LRU Cache), and more. Let's look at a classic Leetcode problem to illustrate the use of queues.

#### Example: Binary Tree Level Order Traversal (Leetcode 102)

Given a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque

def levelOrder(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.popleft()
            if node:
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        
        result.append(level)

    return result

# Usage example
# Assume a binary tree is defined here
# print(levelOrder(root))
```

In this solution, a `deque` from the `collections` module is used as a queue to hold nodes at each level of the tree. The while loop continues as long as there are nodes to process, and for each loop iteration, it processes nodes that are at the same depth level. This ensures a level-by-level traversal of the tree, aligning perfectly with the FIFO nature of queues.

### Comments and Teaching Points

- **Choosing the Right Data Structure:** The choice of `deque` over a list for queue operations is due to its efficient append and pop operations from both ends.
- **Level-by-Level Processing:** The use of a loop within a loop allows for processing all nodes at a given depth before moving on to the nodes at the next depth, which is crucial for level order traversal in trees.

Queues offer a versatile tool in algorithm design, particularly for problems requiring sequential processing or breadth-first search. Their implementation and application can greatly simplify the solution to complex problems, making them a vital concept for software interview preparation.

# Linked List

Linked lists are a fundamental data structure in computer science, widely used for their flexibility and efficiency in certain types of operations. A linked list is a collection of nodes, where each node contains data and a reference (or link) to the next node in the sequence. This structure allows for efficient insertion and deletion of elements, as these operations do not require the data to be contiguous in memory.

### Types of Linked Lists

- **Singly Linked List:** Each node has data and a reference to the next node.
- **Doubly Linked List:** Each node has data and two references—one to the next node and one to the previous node, allowing for traversal in both directions.
- **Circular Linked List:** Similar to a singly or doubly linked list, but the last node references the first node, creating a circular structure.

### Advantages of Linked Lists

- **Dynamic Size:** Unlike arrays, linked lists can grow and shrink in size without the need for reallocation or resizing.
- **Efficient Insertions/Deletions:** Adding or removing elements from the beginning or middle of the list does not require shifting elements, as in the case of arrays.

### Disadvantages of Linked Lists

- **Random Access:** Direct access to an element (e.g., via index) is not possible. One must traverse the list from the beginning to reach a specific element, which can be inefficient.
- **Memory Overhead:** Each node requires additional memory for the reference (pointer) in addition to the data.

### Example in Python: Implementing a Singly Linked List

Here's a basic implementation of a singly linked list in Python 3, demonstrating how to define a node, insert elements, and traverse the list.

```python
class ListNode:
    def __init__(self, value=0, next=None):
        self.value = value
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None  # The list is initially empty

    def insert_at_head(self, value):
        # Create a new node with the given value and set it as the new head of the list
        new_node = ListNode(value)
        new_node.next = self.head
        self.head = new_node

    def print_list(self):
        # Traverse the list and print each node's value
        current_node = self.head
        while current_node:
            print(current_node.value, end=' -> ')
            current_node = current_node.next
        print('None')

# Usage
linked_list = LinkedList()
linked_list.insert_at_head(3)
linked_list.insert_at_head(2)
linked_list.insert_at_head(1)
linked_list.print_list()
```

This example demonstrates the basics of working with linked lists, including node creation, list traversal, and insertion at the head of the list.

### Real-World Example: Undo Functionality in Applications

A common real-world use of linked lists is to implement undo functionality in applications. Each node in the linked list can represent a state of the document or application. When the user makes a change, a new state is added to the list. To undo an action, the application can revert to the previous node's state. This is efficient because each state change doesn't require copying the entire document's state, just the differences.

### Conclusion

Linked lists are a versatile and essential data structure, particularly useful where efficient insertions and deletions are crucial. While they come with trade-offs such as lack of random access and additional memory overhead for pointers, their benefits often make them the data structure of choice for certain problems and scenarios in software development.

# Binary Trees
Binary trees are a foundational concept in computer science, used to model hierarchical data structures. They consist of nodes connected by edges, where each node contains a value and pointers to two child nodes, conventionally referred to as the left child and the right child. The topmost node is called the root of the tree. A binary tree is characterized by the fact that each node can have at most two children, which differentiates it from other types of trees where a node could have any number of children.

### Key Properties:
- **Depth of a Node**: The number of edges from the root to the node.
- **Height of a Tree**: The number of edges on the longest downward path between the root and a leaf.
- **Full Binary Tree**: Every node other than the leaves has two children.
- **Complete Binary Tree**: All levels are fully filled except possibly the last level, which is filled from left to right.
- **Balanced Binary Tree**: The height of the two subtrees of any node differ by no more than one.
- **Binary Search Tree (BST)**: A special kind of binary tree where the left child node is less than the parent node, and the right child node is greater than the parent node.
- **Perfect Binary Tree**: A perfect binary tree is a type of binary tree in which every internal node has exactly two children, and all leaf nodes are at the same depth or level. This means it's both "full" and "complete." In a perfect binary tree, there are exactly $2^k - 1$ nodes (where $k$ is the number of levels).

### Visualizing Differences:
- **Full Binary Tree** can have some leaves at different levels, but every node must either have 2 or no children.
- **Complete Binary Tree** ensures that all levels are filled except possibly the last one, which is filled from left to right.
- **Balanced Binary Tree** focuses on ensuring that the height difference between left and right subtrees is no more than one, without strict requirements on how each level is filled.
- **Binary Search Tree (BST)** prioritizes the order of elements (left < parent < right) without imposing structural completeness.
- **Perfect Binary Tree** combines the fullness of a "Full Binary Tree" with the level completion aspect, ensuring every level is completely filled, making it symmetrical.

![Different Types of Binary Trees](./images/comparison_of_binary_trees.png)
*<small>Illustration of Different Types of Binary Trees.</small>*

### Examples:

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Example of creating a simple binary tree
#       1
#      / \
#     2   3
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
```

### Real World Implications:
Binary trees are crucial in many computing algorithms and systems. They're used in database indexes for efficient data retrieval, in sorting algorithms like heapsort (via binary heaps), and in decision-making processes such as those found in machine learning decision trees. Binary search trees, a subtype of binary trees, are especially useful for searching and sorting operations due to their ability to reduce the search space by half at each step.

### Fundamental LeetCode Problems:

1. **Invert a Binary Tree (LeetCode #226)**

The problem involves flipping a binary tree around its center, meaning the left child becomes the right child and vice versa for every node in the tree.

- **Solution Approach**: A recursive strategy works well here. For each node, we swap its left and right children, then proceed to invert the left and right subtrees recursively.

```python
def invertTree(root):
    if not root:
        return None
    # Swap the left and right child
    root.left, root.right = root.right, root.left
    # Recursively invert the subtrees
    invertTree(root.left)
    invertTree(root.right)
    return root
```

- **Time Complexity**: O(n), where n is the number of nodes, since we visit each node exactly once.
- **Space Complexity**: O(h), where h is the height of the tree. This space is used by the call stack during the recursion.

2. **Maximum Depth of Binary Tree (LeetCode #104)**

This problem requires finding the maximum depth (or height) of a binary tree, which is the longest path from the root node down to the farthest leaf node.

- **Solution Approach**: We can solve this using recursion by computing the height of the left and right subtrees. The maximum depth at any node will be the max depth of its subtrees plus one (for the current node).

```python
def maxDepth(root):
    if not root:
        return 0
    # Recursively find the depth of the left and right subtrees
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    # The depth of the current node is max of left and right depths plus one
    return max(left_depth, right_depth) + 1
```

- **Time Complexity**: O(n), as we need to visit each node.
- **Space Complexity**: O(h), due to the recursion stack, where h is the height of the tree.

Through these examples, we see the elegance and efficiency binary trees bring to solving complex problems, highlighting their importance in software development and algorithm design.

# Depth-First Search (DFS) in Binary Trees
Binary Tree Depth First Search (DFS) is a fundamental algorithmic technique used to explore and process all the nodes in a binary tree. Unlike Breadth-First Search (BFS) that explores the tree level-by-level, DFS goes as deep as possible down one path before backing up and trying another. In the context of binary trees, this means moving through the tree by visiting a node's child before visiting its sibling. DFS is particularly useful for tasks that need to explore all possible paths or need to process a tree in a specific order (preorder, inorder, or postorder).

### Variants of DFS in Binary Trees

There are three primary ways to perform DFS in a binary tree:

1. **Preorder Traversal**: Visit the current node before its children. The process follows the sequence: Visit -> Go Left -> Go Right.
2. **Inorder Traversal**: Visit the left child, then the current node, and finally the right child. This sequence: Go Left -> Visit -> Go Right results in visiting nodes in ascending order in a binary search tree.
3. **Postorder Traversal**: Visit the current node after its children. The sequence is: Go Left -> Go Right -> Visit.

### Examples

To make these concepts clear, let's consider a binary tree:

```
    A
   / \
  B   C
 / \   \
D   E   F
```

- **Preorder Traversal**: A -> B -> D -> E -> C -> F
- **Inorder Traversal**: D -> B -> E -> A -> C -> F
- **Postorder Traversal**: D -> E -> B -> F -> C -> A

### Real-world Implications

In real-world applications, DFS is invaluable for hierarchical data structures and scenarios like:
- **Web Crawling**: Where a DFS approach can explore a website's links deeply before moving to adjacent links.
- **Solving Puzzles**: Such as mazes, where DFS can explore each possible path to completion before backtracking.
- **Dependency Resolution**: In systems like package managers where dependencies must be installed before the package that requires them.

### LeetCode Problems

Let's apply DFS to solve two fundamental LeetCode problems:

1. **Maximum Depth of Binary Tree** (LeetCode Problem 104): Find the maximum depth of a binary tree.
2. **Path Sum** (LeetCode Problem 112): Determine if the tree has a root-to-leaf path such that adding up all the values along the path equals a given sum.

I'll now solve these problems with detailed, commented Python code to demonstrate DFS in action.

#### Problem 1: Maximum Depth of Binary Tree

First, let's tackle finding the maximum depth of a binary tree.

```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepth(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    if not root:
        return 0  # Base case: if the node is null, depth is 0
    
    # Recursive DFS on left and right subtrees to find their depth
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)
    
    # The depth of the current node is max of left and right subtree depths + 1
    return max(left_depth, right_depth) + 1
```

#### Problem 2: Path Sum

Next, let's solve the problem of checking if a tree has a root-to-leaf path with a given sum.

```python
def hasPathSum(root, sum):
    """
    :type root: TreeNode
    :type sum: int
    :rtype: bool
    """
    if not root:
        return False  # Base case: if the node is null, it can't contribute to the sum
    
    # Check if it's a leaf node and the path sum matches the required sum
    if not root.left and not root.right and root.val == sum:
        return True
    
    # Subtract the current node's value from sum and recursively check left and right subtrees
    sum -= root.val
    return hasPathSum(root.left, sum) or hasPathSum(root.right, sum)
```

These solutions exemplify how DFS can be applied to binary trees to solve complex problems efficiently. The time complexity for both problems is $O(N)$, where $N$ is the number of nodes in the tree, as we potentially visit each node once. The space complexity is $O(H)$, where $H$ is the height of the tree, due to the call stack during the recursion, which in the worst case can be $O(N)$ for a skewed tree but is generally $O(log N)$ for a balanced tree.

## When to use Preorder, Inorder, or Postorder?

Understanding when to use preorder, inorder, and postorder traversals in depth-first search (DFS) of binary trees is foundational for solving various types of problems. Each traversal order offers a unique approach to exploring the nodes of a binary tree, and selecting the right one depends on the specific requirements of the problem you're trying to solve.

### Preorder Traversal (Root, Left, Right)
Preorder traversal is used when you need to explore roots before inspecting leaves. It's useful in problems where you need to replicate the tree structure or when the process of visiting a node includes operations that depend on information from the parent node.

**Real-world implication**: Imagine a filesystem where directories and files are structured as a binary tree. Preorder traversal could be used to copy the filesystem, where you need to create a directory before you can create its subdirectories and files.

**LeetCode Example**: [LeetCode Problem 144 - Binary Tree Preorder Traversal](https://leetcode.com/problems/binary-tree-preorder-traversal/)

#### Solution:
```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def preorderTraversal(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    
    # The preorder traversal list
    traversal = []
    
    # Define a recursive function to perform preorder traversal
    def preorder(node):
        if not node:
            return
        traversal.append(node.val)  # Visit the root
        preorder(node.left)         # Traverse left subtree
        preorder(node.right)        # Traverse right subtree
    
    preorder(root)
    return traversal
```

### Inorder Traversal (Left, Root, Right)
Inorder traversal is particularly useful for binary search trees (BST), where it returns nodes in non-decreasing order. This property makes inorder traversal ideal for problems that require sorted data from a BST.

**Real-world implication**: For a BST representing a sequence of events ordered by time, inorder traversal can list the events in chronological order.

**LeetCode Example**: [LeetCode Problem 94 - Binary Tree Inorder Traversal](https://leetcode.com/problems/binary-tree-inorder-traversal/)

#### Solution:
```python
def inorderTraversal(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    
    traversal = []
    
    def inorder(node):
        if not node:
            return
        inorder(node.left)         # Traverse left subtree
        traversal.append(node.val)  # Visit the root
        inorder(node.right)        # Traverse right subtree
    
    inorder(root)
    return traversal
```

### Postorder Traversal (Left, Right, Root)
Postorder traversal is used when you need to visit all children nodes before you deal with the node itself. This approach is useful for problems that require a bottom-up solution, such as calculating the height of the tree or deleting the tree.

**Real-world implication**: In a project dependency graph represented as a binary tree, postorder traversal can ensure that dependent tasks are completed before a parent task starts.

**LeetCode Example**: [LeetCode Problem 145 - Binary Tree Postorder Traversal](https://leetcode.com/problems/binary-tree-postorder-traversal/)

#### Solution:
```python
def postorderTraversal(root):
    """
    :type root: TreeNode
    :rtype: List[int]
    """
    if not root:
        return []
    
    traversal = []
    
    def postorder(node):
        if not node:
            return
        postorder(node.left)         # Traverse left subtree
        postorder(node.right)        # Traverse right subtree
        traversal.append(node.val)  # Visit the root
    
    postorder(root)
    return traversal
```

# Binary Tree - Breadth-First Search (BFS)

Let's dive deep into the Binary Tree Breadth-First Search (BFS) pattern, a fundamental and powerful approach to traversing trees.

### 1. Concept and Example

**Breadth-First Search (BFS)** is a traversal technique that explores nodes layer by layer. In the context of a binary tree, BFS starts at the root node, explores all nodes at the current depth (level) before moving on to nodes at the next depth level. This is typically implemented using a queue.

Here’s a step-by-step breakdown of BFS on a binary tree:
1. Initialize a queue and add the root node to it.
2. While the queue is not empty:
   - Dequeue the front node.
   - Process the current node (e.g., print its value).
   - Enqueue the node's children (left first, then right).

**Example:**
Consider the following binary tree:
```
        1
       / \
      2   3
     / \   \
    4   5   6
```
The BFS traversal of this tree would be: 1, 2, 3, 4, 5, 6.

### 2. Real-World Implications

BFS is not just a theoretical construct; it has practical applications in various domains:
- **Network Broadcasting:** In computer networks, BFS can be used to send broadcasts through a network, ensuring all nodes receive the message in the shortest time.
- **Social Networking:** BFS can help in features like "People You May Know," as it starts with direct friends (first level) and then moves to friends of friends.

### 3. Leetcode Problems

Let’s apply BFS to solve two fundamental problems from Leetcode that illustrate its utility in different scenarios.

#### Problem 1: "Binary Tree Level Order Traversal" (Leetcode 102)

**Task:** Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Here's how you can approach this problem using BFS:

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def levelOrder(root):
    if not root:
        return []
    
    result, queue = [], deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(current_level)
    
    return result

# Example Usage
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3, None, TreeNode(6)))
print(levelOrder(root))  # Output: [[1], [2, 3], [4, 5, 6]]
```

**Time Complexity:** \(O(n)\), where \(n\) is the number of nodes in the tree (each node is processed once).
**Space Complexity:** \(O(n)\), to hold the queue and output structure.

#### Problem 2: "Minimum Depth of Binary Tree" (Leetcode 111)

**Task:** Find the minimum depth of a binary tree, which is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Approach with BFS:

```python
def minDepth(root):
    if not root:
        return 0
    
    queue = deque([(root, 1)])  # Node with its depth
    
    while queue:
        node, depth = queue.popleft()
        if not node.left and not node.right:
            return depth  # Return the depth at the first leaf node
        if node.left:
            queue.append((node.left, depth + 1))
        if node.right:
            queue.append((node.right, depth + 1))

# Example usage
root = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
print(minDepth(root))  # Output: 2
```

**Time Complexity:** \(O(n)\), since every node is visited.
**Space Complexity:** \(O(n)\), the worst case for a skewed tree but typically less.

### 4. Visual Representation

A visual might help clarify the BFS process. Let's draw the BFS traversal process on a sample tree:

```python
import networkx as nx
import matplotlib.pyplot as plt

def draw_binary_tree(root):
    G = nx.DiGraph()
    queue = deque([(root, "1")])
    
    while queue:
        node, path = queue.popleft()
        if node.left:
            G.add_edge(node.val, node.left.val)
            queue.append((node

.left, path+"L"))
        if node.right:
            G.add_edge(node.val, node.right.val)
            queue.append((node.right, path+"R"))
            
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=20, font_color='darkred')
    plt.title("Binary Tree Visualization")
    plt.show()

# Visualize the tree
draw_binary_tree(root)
```

By following these steps, we've explored the BFS pattern in depth, provided real-world contexts, tackled representative problems, and visualized the concept. This comprehensive approach helps solidify understanding and application in software interviews and real-world tasks.

# Introduction to Binary Search Trees (BSTs)

A Binary Search Tree (BST) is a type of data structure that organizes data in a way that allows for efficient searching, insertion, and deletion operations. Each node in a BST has at most two children: a left child and a right child. The key feature of a BST is that it maintains a specific order among its elements: for any node in the tree, the values in its left subtree are less than its own value, and the values in its right subtree are greater than its own value. This property ensures that the operations of searching, inserting, and deleting can be performed efficiently, typically in \(O(log n)\) time where \(n\) is the number of nodes in the tree, assuming the tree is balanced.

#### Structure of a BST Node
A typical BST node contains:
- A data field.
- A reference to the left child.
- A reference to the right child.

#### Basic Operations on BST
1. **Search**: To find a value in the tree, start at the root and recursively travel down. Go left if the value is less than the current node's value, and go right if it's greater.
2. **Insert**: To insert a new value, follow the same logic as search to find the correct spot to insert the new node so that the BST property is maintained.
3. **Delete**: To delete a node, find the node, then:
   - If it has no children, simply remove it.
   - If it has one child, replace it with its child.
   - If it has two children, replace it with its in-order successor or predecessor and then delete that node.

### Examples
Consider this BST:

```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
```

- Searching for 6 would involve traversing: 8 (go left) -> 3 (go right) -> 6 (found).
- Inserting 5 would involve traversing: 8 (go left) -> 3 (go right) -> 6 (go left) -> 4 (go right) -> insert 5.
- Deleting 3 (which has two children) would typically involve replacing it with 4 (its in-order successor). Explained more in the next section.


### Deleting node from BST

Deleting a node from a Binary Search Tree (BST), particularly one with two children like the node `3` in your example, follows a specific set of rules to maintain the properties of the BST. Here's how you would delete node `3` from the tree you've provided:

#### Current Tree Structure
```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
```

#### Steps to Delete Node `3`

1. **Identify the node to be deleted**: Node `3` has two children — `1` and `6`.

2. **Find the in-order successor**: The in-order successor of a node in a BST is the smallest node that is larger than the node being deleted. For node `3`, you would look in its right subtree and find the smallest node there. This is done by going to the right child (`6`), and then moving to the leftmost node of this subtree. In this case, `4` (child of `6`) is the in-order successor because it is the leftmost node in the right subtree of `3`.

3. **Replace the value of node `3` with its in-order successor (`4`)**: You substitute `3` with `4`.

4. **Remove the in-order successor node (`4`) from its original position**: Since `4` has been moved up, you now need to remove the original `4`. Node `4` has no left child but may have a right child. Any right child would take the place of `4`.

Here's a step-by-step breakdown of what the tree looks like after each step:

- Before deletion:
  ```
          8
         / \
        3   10
       / \    \
      1   6    14
         / \   /
        4   7 13
  ```

- After replacing `3` with `4`:
  ```
          8
         / \
        4   10
       / \    \
      1   6    14
         / \   /
        x   7 13
  ```

- If `4` had a right child, it would replace `4` at its original position. In this scenario, if `4` had no children, you simply remove `4`.

The final tree structure would look like this:
```
        8
       / \
      4   10
     / \    \
    1   6    14
       / \   /
          7 13
```

This removal and replacement ensure that the BST properties are maintained, where every left child is smaller and every right child is larger than their parent node.

#### In-order Successor with Child
If the node `4` had a child (or children), the steps to delete node `3` from the BST and replace it with `4` would adjust slightly to accommodate the children of `4`. Let's assume `4` had a right child for demonstration purposes:

#### Current Tree Structure with Right Child for `4`
```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
       \
        5
```

#### Steps to Delete Node `3` and Replace with `4` (Assuming `4` has a Right Child `5`)

1. **Identify and Decide on Replacement**: Find the in-order successor of `3`, which is `4`.

2. **Replace `3` with `4`**: Move `4` to where `3` was.

3. **Handle the Children of `4`**: Since `4` has a right child (`5`), this child must be reconnected to maintain the BST properties.

#### Adjusting the Tree

- Before replacement:
  ```
        8
       / \
      3   10
     / \    \
    1   6    14
       / \   /
      4   7 13
       \
        5
  ```

- After replacing `3` with `4` and handling `5`:
  ```
        8
       / \
      4   10
     / \    \
    1   6    14
       / \   /
       5  7 13
  ```

Here, after `4` replaces `3`, `5` is reconnected as the left child of `6`. This reconnection is crucial because `5` is less than `6` and fits appropriately into the left child position.

### Final Tree Structure
```
        8
       / \
      4   10
     / \    \
    1   6    14
       / \   /
       5  7 13
```
This series of steps ensures that the structure and properties of the BST are properly maintained after the deletion of `3` and the repositioning of its in-order successor `4`, along with the proper placement of `4`'s children.

## Time Complexity

When preparing for technical interviews, understanding the time complexities associated with various operations on a Binary Search Tree (BST) is crucial. Here’s a general overview of the time complexities for common BST operations:

### 1. **Search**
- **Average Case**: O(log n)
- **Worst Case**: O(n)
  
**Explanation**: In a balanced BST, the depth is approximately log₂n, making the average case time complexity O(log n). However, in an unbalanced tree, such as when the nodes are inserted in a sorted order, the tree can degrade to a linked list with a worst-case time complexity of O(n).

### 2. **Insertion**
- **Average Case**: O(log n)
- **Worst Case**: O(n)

**Explanation**: Similar to search, insertion in a balanced BST will take O(log n) time, as each comparison allows the operations to skip about half of the tree. However, like search, in the worst case where the tree becomes unbalanced, the time complexity can degrade to O(n).

### 3. **Deletion**
- **Average Case**: O(log n)
- **Worst Case**: O(n)

**Explanation**: Deletion might require additional steps compared to insertion or search, such as finding an in-order successor for a node with two children. Despite these additional steps, the average time complexity remains O(log n) for balanced trees. However, in an unbalanced tree, it again degrades to O(n).

### 4. **Traversal (In-order, Pre-order, Post-order)**
- **Time Complexity**: O(n)

**Explanation**: Tree traversal techniques like in-order, pre-order, and post-order require visiting every node exactly once. Hence, the time complexity is O(n) regardless of the tree’s balance.

### 5. **Finding Minimum/Maximum**
- **Time Complexity**: O(log n) for balanced, O(n) for unbalanced

**Explanation**: The minimum or maximum value in a BST is found by traversing to the leftmost or rightmost node, respectively. In a balanced tree, this operation takes O(log n) time, while in an unbalanced tree (e.g., when skewed to one side), it could take O(n) time.

### Special Note on Tree Balance
- **Self-Balancing BSTs**: Structures like AVL Trees and Red-Black Trees maintain balance through rotations and other operations to ensure that the tree remains balanced after each insertion or deletion, preserving the O(log n) time complexity for all main operations.

#### AVL Trees
- **Balancing Criterion**: AVL Trees maintain balance by ensuring that the heights of the two child subtrees of any node differ by no more than one. After each insertion or deletion, AVL trees use rotations (single or double) to re-balance the tree if this height condition is violated.

#### Red-Black Trees
- **Balancing Criterion**: These trees use an additional set of properties involving node colors (red or black) along with specific rules regarding the colors of node parents and children. After every insertion and deletion, certain operations are performed to repaint nodes and perform rotations to maintain the tree's balance, ensuring that the tree height remains logarithmic in relation to the number of nodes.

#### Why This Matters
- **Time Complexity**: The primary advantage of self-balancing trees is that they maintain O(log n) time complexity for search, insert, and delete operations by ensuring the tree height stays balanced.

### Importance in Interviews
In interviews, it's beneficial to not only know these complexities but also to be able to discuss ways to optimize BST performance, such as using self-balancing trees. Demonstrating knowledge about potential worst-case scenarios and how to avoid them can also be particularly impressive to interviewers.

### Real-World Applications
BSTs are useful in many applications where data needs to be frequently searched, inserted, or deleted. They are used in:
- Implementing databases and file systems where quick search, insertion, and deletion are necessary.
- Game development for storing objects in a world and quickly querying their positions.

### Common Leetcode Problems
1. **Validate Binary Search Tree (Leetcode 98)**: Determine if a binary tree is a binary search tree.
2. **Lowest Common Ancestor of a Binary Search Tree (Leetcode 235)**: Find the lowest common ancestor of two nodes in a BST.

Now, let's dive into the detailed solutions of these two Leetcode problems to understand how we can implement and manipulate BSTs in practice.

#### 1. Validate Binary Search Tree (Leetcode 98)

**Problem Statement**:
Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST is defined as follows:
- The left subtree of a node contains only nodes with keys less than the node's key.
- The right subtree of a node contains only nodes with keys greater than the node's key.
- Both the left and right subtrees must also be binary search trees.

**Solution and Explanation**:
We'll use recursion to validate the BST by checking at each step if the node's value is within valid ranges which get updated as we move left (upper bound gets tighter) or right (lower bound gets tighter).

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_valid_bst(root, low=float('-inf'), high=float('inf')):
    # Base case: An empty tree is a BST
    if not root:
        return True
    
    # If current node's value does not fall within the valid range, return False
    if not (low < root.val < high):
        return False
    
    # Recursively validate the left and right subtree
    # Update the ranges accordingly:
    # Left subtree must have values < root.val
    # Right subtree must have values > root.val
    return (is_valid_bst(root.left, low, root.val) and
            is_valid_bst(root.right, root.val, high))

# Example Usage:
# Constructing a simple BST:
#       2
#      / \
#     1   3
node1 = TreeNode(1)
node3 = TreeNode(3)
root = TreeNode(

2, node1, node3)

# Should return True as this is a valid BST
print(is_valid_bst(root))
```

This function will check every node in the tree ensuring it obeys the constraints of BST with respect to its position. It does this efficiently by narrowing the valid range of values as it traverses the tree, ensuring a time complexity of \(O(n)\), where \(n\) is the number of nodes, since each node is visited once.


