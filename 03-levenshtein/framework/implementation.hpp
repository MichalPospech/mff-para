#ifndef LEVENSHTEIN_IMPLEMENTATION_HPP
#define LEVENSHTEIN_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <vector>

#include <utility>
#include <algorithm>

#define BLOCK_SIZE 1024

template <typename C = char, typename DIST = std::size_t, bool DEBUG = false>
class EditDistance : public IEditDistance<C, DIST, DEBUG>
{
private:
	std::vector<DIST> distances;
	DIST len1;
	DIST len2;
	std::size_t num_diags;
	std::size_t num_blocks1;
	std::size_t num_blocks2;
	std::size_t min_blocks;
	std::size_t max_blocks;

private:
	void calculate_submatrix(const C *str1, const C *str2, std::size_t x1, std::size_t x2, std::size_t y1, std::size_t y2, std::size_t start_index)
	{

		std::size_t x_size = x2 - x1;
		std::size_t y_size = y2 - y1;
		for (size_t x = 0; x < x_size; x++)
		{
#pragma omp simd
			for (size_t y = 0; y < y_size; y++)
			{
				std::size_t sub_cost = (str1[x1 + x] == str2[y1 + y]) ? 0 : 1;
				std::size_t diag_index = start_index + y_size - y + x;
				distances[diag_index] = std::min({distances[diag_index] + sub_cost, distances[diag_index - 1] + 1, distances[diag_index + 1] + 1});
			}
		}
	}

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param len1, len2 Lengths of first and second string respectively.
	 */
	virtual void init(DIST len1, DIST len2)
	{
		if (len1 < len2)
		{
			std::swap(len1, len2);
		}
		this->len1 = len1;
		this->len2 = len2;
		num_blocks1 = len1 / BLOCK_SIZE;
		num_blocks2 = len2 / BLOCK_SIZE;
		num_diags = num_blocks1 + num_blocks2 - 1;
		distances.resize(len1 + len2 + 1);
		for (size_t i = len2 + 1; i < len2 + len1 + 2; i++)
		{
			distances[i] = i - len2;
		}
		for (size_t i = 0; i < len2; i++)
		{
			distances[i] = len2 - i;
		}
		max_blocks = (num_blocks1 > num_blocks2) ? num_blocks1 : num_blocks2;
		min_blocks = (num_blocks1 < num_blocks2) ? num_blocks1 : num_blocks2;
	};

	/*
	 * \brief Compute the distance between two strings.
	 * \param str1, str2 Strings to be compared.
	 * \result The computed edit distance.
	 */

	virtual DIST compute(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		const C *s1 = &str1[0];
		const C *s2 = &str2[0];
		if (str1.size() < str2.size())
			std::swap(s1, s2);
		for (std::size_t i = 0; i < num_diags; i++)
		{
			std::size_t diag_num_blocks = (i < min_blocks) ? (i + 1) : ((i < max_blocks) ? min_blocks : (num_diags - i));
#pragma omp parallel for schedule(static)
			for (std::size_t j = 0; j < diag_num_blocks; j++)
			{

				std::size_t x_index = (i < min_blocks) ? j : (i - min_blocks + 1 + j);
				std::size_t y_index = (i < min_blocks) ? (i - j) : (min_blocks - 1 - j);
				std::size_t x1 = x_index * BLOCK_SIZE;
				std::size_t y1 = y_index * BLOCK_SIZE;
				std::size_t x2 = x1 + BLOCK_SIZE;
				std::size_t y2 = y1 + BLOCK_SIZE;
				std::size_t start_index = len2 - y2 + x1;
				calculate_submatrix(s1, s2, x1, x2, y1, y2, start_index);
			}
		}

		return distances[len1];
	}
};
#endif
