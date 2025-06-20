// filepath: /Users/adithyakomandur/StudioProjects/HackDay 2025/Hack-Day-2025/UI/swoosh-realm/src/lib/utils.ts
export function cn(...classes: (string | undefined | false | null)[]) {
  return classes.filter(Boolean).join(" ");
}