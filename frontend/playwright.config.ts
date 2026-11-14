import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:5173", trace: "retain-on-failure" },
  projects: [
    {
      name: "http",
      testMatch: "ui.spec.ts",
      use: { ...devices["Desktop Chrome"], channel: "msedge" },
    },
    {
      name: "mock",
      testMatch: "mock.spec.ts",
      use: {
        ...devices["Desktop Chrome"],
        channel: "msedge",
        baseURL: "http://127.0.0.1:5174",
      },
    },
  ],
  webServer: [
    {
      command: "npm run dev -- --host 127.0.0.1",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: false,
      env: { VITE_USE_MOCK_API: "false" },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5174",
      url: "http://127.0.0.1:5174",
      reuseExistingServer: false,
      env: { VITE_USE_MOCK_API: "true" },
    },
  ],
});
