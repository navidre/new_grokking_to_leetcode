## Note
All the matrials in this page are generated using the following custom GPT:

🚀 **[Software Interview Mentor](https://chat.openai.com/g/g-n76b8bWJo-software-interview-mentor)** - Learn about different concepts! 🤖 (Requires ChatGPT Plus)

## Sliding Window

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
