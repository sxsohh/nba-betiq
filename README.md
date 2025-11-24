"use client";

import Image from "next/image";
import React, { useEffect, useState } from "react";

const PROFILE = {
  name: "Stefan Soh",
  title: "Student-Athlete · Computer Science & Physics",
  email: "stefanxsoh@gmail.com",
  linkedin: "https://www.linkedin.com/in/stefanxsoh/",
  github: "https://github.com/sxsohh",
  resume: "/StefanSohResume.pdf",
  scholarshipVideo: "https://www.youtube.com/embed/lsdA_NqvzGw",
};

type ProjectLink = { label: string; href: string };

type Project = {
  title: string;
  year: string;
  description: string;
  tags: string[];
  links?: ProjectLink[];
};

const PROJECTS: Project[] = [
  {
    title: "NBA BetIQ · Sports Betting House Edge Analysis",
    year: "2025",
    description:
      "A FastAPI and machine learning project that exposes how sportsbooks create a constant hidden edge through moneylines, spreads, and totals. This project lets users simulate betting strategies and visualize long-term losses to highlight the growing sports betting epidemic.",
    tags: ["Python", "FastAPI", "Machine Learning", "Data Science"],
    links: [
      { label: "View Project Repo", href: "https://github.com/sxsohh/nba-betiq" },
    ],
  },
  {
    title: "Late Game Fouling Computer Vision Project",
    year: "2025",
    description:
      "Computer vision system that identifies players on the floor, matches them to free throw percentages, and helps coaches decide who to foul in late game situations. This project earned a 20K scholarship.",
    tags: ["OpenCV", "Python", "Computer Vision", "Basketball"],
    links: [{ label: "Watch Project Video", href: PROFILE.scholarshipVideo }],
  },
  {
    title: "SohAI · AI Phone Agents for Local Businesses",
    year: "2025",
    description:
      "My AI marketing and automation studio that builds HIPAA safe phone agents and workflows for dentists, orthodontists, and other service businesses. Focused on reducing missed calls and automating FAQs.",
    tags: ["Entrepreneurship", "LLM", "Automation"],
    links: [{ label: "Visit SohAI", href: "https://sohai.info" }],
  },
];

export default function Home() {
  const [projects, setProjects] = useState<Project[]>(PROJECTS);
  const [projectQuery, setProjectQuery] = useState("");

  useEffect(() => {
    fetch("/projects.json")
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setProjects(data);
      })
      .catch(() => {});
  }, []);

  const filteredProjects = projects.filter((p) => {
    const q = projectQuery.toLowerCase();
    if (!q) return true;
    return (
      p.title.toLowerCase().includes(q) ||
      p.description.toLowerCase().includes(q) ||
      p.tags.some((t) => t.toLowerCase().includes(q))
    );
  });

  return (
    <main className="min-h-screen">
      
      {/* HERO */}
      <section className="hero-gradient">
        <div className="mx-auto max-w-6xl px-6 py-14 sm:py-20 grid md:grid-cols-2 gap-10 items-center">
          
          {/* Left side */}
          <div>
            <span className="badge">Portfolio</span>
            <h1 className="mt-4 text-5xl sm:text-6xl font-extrabold">
              {PROFILE.name}
            </h1>
            <p className="mt-4 text-lg sm:text-xl">
              I am Stefan, a student athlete and builder working at the
              intersection of basketball, data, and AI. I create systems that
              help coaches, businesses, and communities make better decisions.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <a
                href={PROFILE.resume}
                target="_blank"
                className="rounded-xl bg-white/95 text-[var(--stef-ink)] px-4 py-2 font-semibold shadow hover:bg-white"
              >
                Download Resume
              </a>
              <a
                href={PROFILE.linkedin}
                target="_blank"
                className="rounded-xl bg-white/15 border border-white/40 px-4 py-2 font-semibold hover:bg-white/20"
              >
                LinkedIn
              </a>
              <a
                href={PROFILE.github}
                target="_blank"
                className="rounded-xl bg-white/15 border border-white/40 px-4 py-2 font-semibold hover:bg-white/20"
              >
                GitHub
              </a>
              <a
                href={`mailto:${PROFILE.email}`}
                className="rounded-xl bg-white/15 border border-white/40 px-4 py-2 font-semibold hover:bg-white/20"
              >
                Email Me
              </a>
            </div>
          </div>

          {/* Right side */}
          <div className="justify-self-center">
            <div className="polaroid rotate-2">
              <Image
                src="/b2d57fba-3143-420a-82bd-79fc4a804852.jpg"
                alt="Stefan on the court"
                width={420}
                height={520}
                className="rounded-md object-cover"
                priority
              />
            </div>
          </div>
        </div>
      </section>

      {/* SCHOLARSHIP VIDEO PROJECT */}
      <section className="mx-auto max-w-6xl px-6 -mt-8 pb-10">
        <div className="panel p-6">
          <h2 className="text-2xl font-bold mb-3 text-[var(--stef-ink)]">
            Late Game Fouling Computer Vision Project
          </h2>
          <p className="mb-3">
            A real time tool that identifies players on the floor and pulls
            their free throw data to guide fouling strategy. This project earned
            a 20K scholarship and reflects my focus on practical solutions that
            combine analytics, game feel, and implementation detail.
          </p>

          <div className="relative w-full mt-5" style={{ paddingTop: "56.25%" }}>
            <iframe
              className="absolute left-0 top-0 h-full w-full rounded-lg border-2 border-[var(--stef-ice)]"
              src={PROFILE.scholarshipVideo}
              title="Scholarship / Computer Vision Project Video"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              allowFullScreen
            />
          </div>
        </div>
      </section>

      {/* PROJECTS */}
      <section className="mx-auto max-w-6xl px-6 pb-10">
        <div className="panel p-6">
          <div className="flex flex-col sm:flex-row sm:items-center gap-3">
            <h2 className="text-2xl font-bold text-[var(--stef-ink)]">
              Projects
            </h2>
            <div className="grow" />
            <input
              placeholder="Search projects..."
              className="polaroid p-2 w-full sm:w-80"
              value={projectQuery}
              onChange={(e) => setProjectQuery(e.target.value)}
            />
          </div>

          <hr className="my-4" />

          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {filteredProjects.map((p) => (
              <div key={p.title} className="panel overflow-hidden">
                <div className="p-4">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-[var(--stef-ink)]">
                      {p.title}
                    </h3>
                    <span className="text-xs text-gray-500">{p.year}</span>
                  </div>
                  <p className="mt-2 text-sm">{p.description}</p>

                  <div className="mt-3 flex flex-wrap gap-2">
                    {p.tags.map((t) => (
                      <span key={t} className="badge">
                        {t}
                      </span>
                    ))}
                  </div>

                  {p.links && (
                    <div className="mt-3 flex flex-wrap gap-3">
                      {p.links.map((l) => (
                        <a
                          key={l.label}
                          href={l.href}
                          target="_blank"
                          className="underline text-[var(--stef-ink)]"
                        >
                          {l.label}
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* GALLERY */}
      <section className="mx-auto max-w-6xl px-6 pb-14">
        <h2 className="text-2xl font-bold mb-4 text-[var(--stef-ink)]">
          Gallery
        </h2>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <div className="panel overflow-hidden">
            <Image
              src="/5Y2A6586_Original.JPG"
              alt="gallery image"
              width={900}
              height={700}
              className="w-full h-64 object-cover"
            />
          </div>

          <div className="panel overflow-hidden">
            <Image
              src="/4B59FFD2-E6BF-4F06-BDE6-2772AF5ADCF9.jpg"
              alt="gallery image"
              width={900}
              height={700}
              className="w-full h-64 object-cover"
            />
          </div>

          <div className="panel overflow-hidden">
            <Image
              src="/IMG_5782.JPG"
              alt="gallery image"
              width={900}
              height={700}
              className="w-full h-64 object-cover"
            />
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer className="mx-auto max-w-6xl px-6 pb-10 text-sm">
        <div className="panel p-4 text-center">
          © {new Date().getFullYear()} {PROFILE.name} ·{" "}
          <a className="underline" href={`mailto:${PROFILE.email}`}>
            {PROFILE.email}
          </a>{" "}
          ·{" "}
          <a className="underline" href="https://sohai.info" target="_blank">
            SohAI
          </a>
        </div>
      </footer>
    </main>
  );
}
