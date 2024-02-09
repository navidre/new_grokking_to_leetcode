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
