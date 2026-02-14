"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

interface Props {
  href: string;
  children: React.ReactNode;
  onClick?: () => void;
  exact?: boolean;
}

export default function NavLink({ href, children, onClick, exact }: Props) {
  const pathname = usePathname();
  const isActive = exact
    ? pathname === href
    : href === "/"
      ? pathname === "/"
      : pathname.startsWith(href);

  return (
    <Link
      href={href}
      onClick={onClick}
      className={`transition-colors ${
        isActive
          ? "text-blue-700 font-semibold"
          : "text-gray-600 hover:text-gray-900"
      }`}
    >
      {children}
    </Link>
  );
}
