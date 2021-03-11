using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Linq;

namespace dns_netcore
{
    class RecursiveResolver : IRecursiveResolver
    {
        private IDNSClient dnsClient;

        private ConcurrentDictionary<string, IP4Addr> cache;
        public RecursiveResolver(IDNSClient client)
        {
            this.dnsClient = client;
            cache = new ConcurrentDictionary<string, IP4Addr>();
        }

        public Task<IP4Addr> ResolveRecursive(string domain)
        {
            /*
			 * Just copy-pasted code from serial resolver.
			 * Replace it with your implementation.
			 * Also you may change this method to async (it will work with the interface).
			 */

            return Task.Run(() =>
            { 
                var baseResolver = dnsClient.GetRootServers().First();

                var subdomains = DomainToSubdomains(domain).Reverse();
                var tasks = subdomains.Select(async sd =>
                {

                    var result = await CacheLookup(sd.Domain);
                    return new
                    {
                        Addr = result,
                        Domain = sd
                    };
                });

                var results = Task.WhenAll(tasks).Result;
                var incorrect = results.TakeWhile(r => r.Addr == null).Select(r => r.Domain).Reverse();
                var resolver = results.Select(r => r.Addr).FirstOrDefault(addr => addr != null) ?? baseResolver; 

                foreach (var sub in incorrect)
                {
                    var t = dnsClient.Resolve(resolver, sub.LowestLevel);
                    resolver = t.Result;
					cache.TryAdd(sub.Domain, resolver);
                }
                return resolver;
            });
        }

        private Task<IP4Addr?> CacheLookup(string domain)
        {
            return Task.Run(() =>
            {
                var present = cache.TryGetValue(domain, out var addr);
                if (present)
                {
                    var reverse = dnsClient.Reverse(addr).Result;
                    if (reverse != domain)
                    {
                        cache.TryRemove(domain, out var _);
                    }
                    return reverse == domain ? addr : null;
                }
                return (IP4Addr?)null;
            });
        }

        private IEnumerable<Subdomain> DomainToSubdomains(string domain)
        {
            var split = domain.Split('.');
            Subdomain sub = null;
            foreach (var subdomain in split.Reverse())
            {
                if (sub != null)
                {
                    sub = new Subdomain
                    {
                        LowestLevel = subdomain,
                        Domain = $"{subdomain}.{sub.Domain}"
                    };
                    yield return sub;
                    continue;
                }
                sub = new Subdomain
                {
                    Domain = subdomain,
                    LowestLevel = subdomain
                };
                yield return sub;
            }
        }

        private class Subdomain
        {
            public string LowestLevel { get; set; }
            public string Domain { get; set; }
        }
    }
}