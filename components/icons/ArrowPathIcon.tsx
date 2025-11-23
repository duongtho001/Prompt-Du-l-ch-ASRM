import React from 'react';

const ArrowPathIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    strokeWidth={1.5}
    stroke="currentColor"
    className={className}
    aria-hidden="true"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M16.023 9.348h4.992v-.001a.75.75 0 01.75.75c0 .414-.336.75-.75.75h-4.992a2.25 2.25 0 01-2.25-2.25V6.108a.75.75 0 01.75-.75c.414 0 .75.336.75.75v3.24z"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M4.032 15.652L4.032 15.652a2.25 2.25 0 012.25-2.25h4.992a.75.75 0 010 1.5H6.282a.75.75 0 00-.75.75v3.24a.75.75 0 01-1.5 0v-3.24a2.25 2.25 0 01.22-1.002z"
    />
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      d="M18.364 5.636a9 9 0 010 12.728m-12.728 0a9 9 0 010-12.728m12.728 0L18.364 5.636m-12.728 0L5.636 5.636"
    />
  </svg>
);

export default ArrowPathIcon;
