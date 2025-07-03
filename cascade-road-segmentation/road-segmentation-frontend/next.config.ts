import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'road.sgta.lat',
        port: '',
        pathname: '/api/v1/results/**',
      },
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '8000',
        pathname: '/api/v1/results/**',
      },
    ],
  },
};

export default nextConfig;
