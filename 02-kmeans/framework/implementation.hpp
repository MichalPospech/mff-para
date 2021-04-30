#ifndef KMEANS_IMPLEMENTATION_HPP
#define KMEANS_IMPLEMENTATION_HPP

#include <interface.hpp>
#include <exception.hpp>
#include <tbb/tbb.h>
#include <math.h>
#include <iostream>

template <typename POINT = point_t, typename ASGN = std::uint8_t, bool DEBUG = false>
class KMeans : public IKMeans<POINT, ASGN, DEBUG>
{
	using coord_t = typename POINT::coord_t;
	class cluster

	{

	public:
		std::size_t count;
		POINT total;
		void add(const POINT &point)
		{
			total.x += point.x;
			total.y += point.y;
			++count;
		}

		POINT get_new_centroid(const POINT &original)
		{
			POINT point = original;
			if (count > 0)
			{
				point.x = total.x / (coord_t)count;
				point.y = total.y / (coord_t)count;
			}
			return point;
		}
	};

	std::vector<coord_t> distances;
	std::size_t k;
	std::size_t points;
	tbb::combinable<std::vector<cluster>> clusters;

	coord_t distance(const POINT &point, const POINT &centroid)
	{
		std::int64_t dx = (std::int64_t)point.x - (std::int64_t)centroid.x;
		std::int64_t dy = (std::int64_t)point.y - (std::int64_t)centroid.y;
		return (coord_t)(dx * dx + dy * dy);
	}

	void assign_cluster(const std::vector<POINT> &centroids, const std::vector<POINT> &points, std::size_t point_index, std::vector<ASGN> &assignments)
	{
		auto point = points[point_index];
		coord_t minDist = distance(point, centroids[0]);
		std::size_t nearest = 0;
		for (std::size_t i = 1; i < centroids.size(); ++i)
		{
			coord_t dist = distance(point, centroids[i]);
			if (dist < minDist)
			{
				minDist = dist;
				nearest = i;
			}
		}
		assignments[point_index] = nearest;
		clusters.local()[nearest].add(point);
	}

public:
	/*
	 * \brief Perform the initialization of the functor (e.g., allocate memory buffers).
	 * \param points Number of points being clustered.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 */
	virtual void
	init(std::size_t points, std::size_t k, std::size_t iters)
	{
		clusters.clear();
		clusters = tbb::combinable<std::vector<cluster>>([k]() {
			return std::vector<cluster>(k);
		});
	}

	/*
	 * \brief Perform the clustering and return the cluster centroids and point assignment
	 *		yielded by the last iteration.
	 * \note First k points are taken as initial centroids for first iteration.
	 * \param points Vector with input points.
	 * \param k Number of clusters.
	 * \param iters Number of refining iterations.
	 * \param centroids Vector where the final cluster centroids should be stored.
	 * \param assignments Vector where the final assignment of the points should be stored.
	 *		The indices should correspond to point indices in 'points' vector.
	 */
	virtual void compute(const std::vector<POINT> &points, std::size_t k, std::size_t iters,
						 std::vector<POINT> &centroids, std::vector<ASGN> &assignments)
	{
		centroids.resize(k);
		for (size_t i = 0; i < k; i++)
		{
			centroids[i] = points[i];
		}

		assignments.resize(points.size());
		for (std::size_t i = 0; i < iters; i++)
		{
			clusters.clear();
			tbb::parallel_for(tbb::blocked_range<std::size_t>(0, points.size(), 1024), [&](auto &&range) {
				for (auto i = range.begin(); i != range.end(); ++i)
					assign_cluster(centroids, points, i, assignments);
			});
			auto calculated_clusters = clusters.combine([k](auto v1, auto v2) {
				for (size_t i = 0; i < k; i++)
				{
					v1[i].total.x += v2[i].total.x;
					v1[i].total.y += v2[i].total.y;
					v1[i].count += v2[i].count;
				}
				return v1;
			});
			for (size_t i = 0; i < k; i++)
			{
				if (DEBUG)
					std::cout << i << " " << calculated_clusters[i].count << std::endl;
				centroids[i] = calculated_clusters[i].get_new_centroid(centroids[i]);
			}
		}
	}
};

#endif
