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

The Prefix Sum pattern is a powerful technique in algorithms and data structures, particularly useful in solving problems involving arrays or lists where we need to frequently calculate the sum of elements in a subrange (i.e., from index `i` to `j`). This pattern is especially efficient when you have to perform multiple such sum calculations on the same array, as it significantly reduces the time complexity from O(n) per query to O(1) after an initial setup.

### Concept of Prefix Sum:

The idea is to preprocess the array and create a new array, often called the prefix sum array. In this array, each element at index `i` stores the sum of all elements from the start of the array up to the element at index `i` in the original array. 

### Creating a Prefix Sum Array:

In Python, you can create a prefix sum array as follows:

```python
def create_prefix_sum(arr):
    prefix_sum = [0] * (len(arr) + 1)
    for i in range(1, len(prefix_sum)):
        prefix_sum[i] = prefix_sum[i-1] + arr[i-1]
    return prefix_sum
```

### Using the Prefix Sum Array:

Once you have the prefix sum array, calculating the sum of elements between indices `i` and `j` in the original array is straightforward. You simply subtract the prefix sum at `i-1` from the prefix sum at `j`. 

### Example:

```python
def range_sum(prefix_sum, i, j):
    return prefix_sum[j+1] - prefix_sum[i]
```

### Real-world Example:

Consider a large-scale financial system that needs to calculate the cumulative transaction amounts over different time ranges frequently. Using the prefix sum pattern, the system can quickly provide these calculations without recalculating the sum each time, greatly improving performance and response times.

### Common LeetCode Problems:

1. **Subarray Sum Equals K (LeetCode 560)**: This problem asks to find the total number of continuous subarrays whose sum equals a given number `k`. With the prefix sum approach, you can solve this problem efficiently.

2. **Range Sum Query - Immutable (LeetCode 303)**: This problem is a direct application of the prefix sum concept. You are required to find the sum of elements between indices `i` and `j` repeatedly.

### Implementing LeetCode 560:

Let's implement a solution for "Subarray Sum Equals K":

```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    hash_table = {0: 1}

    for num in nums:
        prefix_sum += num
        # Check if there is a prefix_sum that, when removed from the current prefix_sum, leaves k
        if prefix_sum - k in hash_table:
            count += hash_table[prefix_sum - k]
        
        # Update the hash table with the current prefix sum
        if prefix_sum in hash_table:
            hash_table[prefix_sum] += 1
        else:
            hash_table[prefix_sum] = 1

    return count
```

This code iterates through the array, keeping track of the cumulative sum (`prefix_sum`). It uses a hash table to store the frequency of each prefix sum. The key insight is that if `prefix_sum - k` is in the hash table, it means there is a subarray that sums to `k`. The frequency of `prefix_sum - k` in the hash table tells us how many such subarrays exist up to the current point.

This pattern is a great example of combining space complexity (using additional memory for the prefix sum array or hash table) for significant gains in time complexity, particularly in scenarios with multiple queries on the same data set.
