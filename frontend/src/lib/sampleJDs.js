// frontend/src/lib/sampleJDs.js
// ─────────────────────────────────────────────────────────────────────────────
// Sample job descriptions, keyed by RoleType enum value (see app/models.py).
//
// When the user picks a role in the role-type <select> on Step 2, the matching
// JD here is dropped into the textarea. The textarea remains editable — users
// can adjust the prefilled JD or wipe it and paste their own.
//
// First line of each JD is "Company: <name>" so the company is visible at
// the top of the textarea. Body mirrors real JDs in structure
// (responsibilities, requirements) so analyze_jd / score_facts produce
// realistic output downstream.
//
// `general` is deliberately omitted from this map — picking "General" in the
// dropdown is a no-op on the JD textarea, preserving anything the user has
// already typed (e.g. a custom JD that doesn't fit a specific role).
// ─────────────────────────────────────────────────────────────────────────────

export const SAMPLE_JDS = {
  data_science: `Company: Mastercard

Senior Data Scientist - Mastercard

Our Purpose

Mastercard powers economies and empowers people in 200+ countries and territories worldwide. Together with our customers, we're helping build a sustainable economy where everyone can prosper. We support a wide range of digital payments choices, making transactions secure, simple, smart and accessible. Our technology and innovation, partnerships and networks combine to deliver a unique set of products and services that help people, businesses and governments realize their greatest potential.

Title and Summary

Data Scientist II
Job Description –Data Scientist
Overview
• Are you excited about Data Assets and the value they brings to an organization?
• Are you an evangelist for data driven decision making?
• Are you motivated to be part of a Global Analytics team that builds large scale Analytical Capabilities supporting end users across 6 continents?
• Do you want to be the go-to resource for data analytics in the company?

Role
• Work closely with global & regional teams to architect, develop, and maintain advanced reporting and data visualization capabilities on large volumes of data in order to support analytics and reporting needs across products, markets and services.
• Translate business requirements into tangible solution specifications and high quality, on time deliverables
• Create repeatable processes to support development of modeling and reporting
• Effectively use tools to manipulate large-scale databases, synthesizing data insights. Provide 1st level insights/conclusions/assessments and present findings via Tableau/PowerBI dashboards, Excel and PowerPoint.
• Apply quality control, data validation, and cleansing processes to new and existing data sources
• Build ML & AI capabilities to support business use case

All About You
• Experience in data management, data mining, data analytics, data reporting, data product development and quantitative analysis
• Have worked on ML/Data Science projects
• Financial Institution or a Payments experience a plus
• Experience presenting data findings in a readable and insight driven format. Experience building support decks.
• Advanced SQL coding
• Experience on Platforms/Environments: SQL Server, Microsoft BI Stack
• Experience on SQL Server Integration Services (SSIS), SQL Server Analysis Services (SSAS) and SQL Server Reporting Services (SSRS) will be an added advantage
• Experience in building data models a plus
• Experience with data visualization tools such as Tableau, Domo, PowerBI a plus
• Experience with Hadoop environments, Python, R, WPS, a plus
• Excellent problem solving, quantitative and analytical skills
• In depth technical knowledge, drive and ability to learn new technologies
• Strong attention to detail and quality
• Team player, excellent communication skills
• Must be able to interact with management, internal stakeholders and collect requirements
• Must be able to perform in a team, use judgment and operate under ambiguity

Education
• Bachelor's or Master's Degree in a Computer Science, Information Technology, Engineering, Mathematics, Statistics, M.S./M.B.A. preferred

Additional Competencies
• Excellent English, quantitative, technical, and communication (oral/written) skills
• Analytical/Problem Solving
• Strong attention to detail and quality
• Creativity/Innovation
• Self-motivated, operates with a sense of urgency
• Project Management/Risk Mitigation
• Able to prioritize and perform multiple tasks simultaneously`,

  ml_engineering: `Company: AuxoAI

AI/ML Engineer - AuxoAI

Role Summary
We are in search of top-tier AI Engineers to join our dynamic team of innovators, creators, and visionaries. As an AI Engineer at AuxoAI, you will have the opportunity to work on ground-breaking projects that utilize artificial intelligence, including cutting-edge generative AI technologies, to solve real-world problems and drive business impact. The role provides a unique opportunity for hands-on experience in the field of artificial intelligence, with a specific focus on generative AI techniques and their practical application in real-world business contexts.

AI Engineers would be developing AI powered companions such as Pricing Copilot, Sales Copilot, Customer Experience Copilot etc. The Sales Copilot for example will provides sales insights, automate routine tasks, and assists sales teams in making informed decisions, ultimately improving the overall sales efficiency.

This is an exciting opportunity to work in a fast paced and innovative environment along with a group of world class and entrepreneurial professionals.

Responsibilities
AI engineers will be responsible to design, develop, and implement AI-driven business copilots that enhance the efficiency and effectiveness of organizations
They will be involved in data pre-processing, feature engineering, model training, and optimization of machine learning algorithms
Additionally, AI engineers will work on integrating AI copilots into existing systems, analyze data to provide insights, and collaborate with cross-functional teams to understand and address business requirements
AI engineers will work closely with clients to understand their requirements, tailor solutions to their specific needs, and provide technical guidance and support throughout the project lifecycle
They have to stay up to date with the latest advancements in AI, machine learning, and related technologies, and actively participate in knowledge sharing within the organization
Qualifications
Strong understanding of machine learning algorithms, deep learning frameworks (e.g., TensorFlow, PyTorch), and data analysis techniques
Proficient in at least one programming language commonly used in AI development (e.g., Python, R etc.)
Solid grasp of data structures, algorithms, and software design principles
Excellent problem-solving skills and the ability to think critically and creatively to develop innovative solutions
Strong communication and interpersonal skills, with the ability to collaborate effectively in a team-oriented, fast-paced environment
Previous internship experience or relevant projects in the AI field is a plus`,

  data_science_consultant: `Company: BCG

BCG - Data Science Consultant

As a part of BCG's X team, you will work closely with consulting teams on a diverse range of advanced analytics and engineering topics. You will have the opportunity to leverage analytical methodologies to deliver value to BCG's Consulting (case) teams and Practice Areas (domain) through providing analytical and engineering subject matter expertise. As a Data Engineer, you will play a crucial role in designing, developing, and maintaining data pipelines, systems, and solutions that empower our clients to make informed business decisions. You will collaborate closely with cross-functional teams, including data scientists, analysts, and business stakeholders, to deliver high-quality data solutions that meet our clients' needs.

YOU'RE GOOD AT

Delivering original analysis and insights to case teams, typically owning all or part of an analytics module whilst integrating with a case team. Design, develop, and maintain efficient and robust data pipelines for extracting, transforming, and loading data from various sources to data warehouses, data lakes, and other storage solutions.
Building data-intensive solutions that are highly available, scalable, reliable, secure, and cost-effective using programming languages like Python and PySpark.
Deep knowledge of Big Data querying and analysis tools, such as PySpark, Hive, Snowflake and Databricks.
Broad expertise in at least one Cloud platform like AWS/GCP/Azure.
Working knowledge of automation and deployment tools such as Airflow, Jenkins, GitHub Actions, etc., as well as infrastructure-as-code technologies like Terraform and CloudFormation.
Good understanding of DevOps, CI/CD pipelines, orchestration, and containerization tools like Docker and Kubernetes.
Basic understanding on Machine Learning methodologies and pipelines.
Communicating analytical insights through sophisticated synthesis and packaging of results (including PPT slides and charts) with consultants, collecting, synthesizing, analyzing case team learning & inputs into new best practices and methodologies.

Communication Skills:
Strong communication skills, enabling effective collaboration with both technical and non-technical team members.

Thinking Analytically
You should be strong in analytical solutioning with hands on experience in advanced analytics delivery, through the entire life cycle of analytics. Strong analytics skills with the ability to develop and codify knowledge and provide analytical advice where required.

What You'll Bring

Bachelor's / Master's degree in computer science engineering/technology
At least 2-4 years within relevant domain of Data Engineering across industries and work experience providing analytics solutions in a commercial setting.
Consulting experience will be considered a plus.
Proficient understanding of distributed computing principles including management of Spark clusters, with all included services - various implementations of Spark preferred.
Basic hands-on experience with Data engineering tasks like productizing data pipelines, building CI/CD pipeline, code orchestration using tools like Airflow, DevOps etc.

Good to have:
Software engineering concepts and best practices, like API design and development, testing frameworks, packaging etc.
Experience with NoSQL databases, such as HBase, Cassandra, MongoDB
Knowledge on web development technologies.
Understanding of different stages of machine learning system design and development`,

  quant_finance: `Company: Tower Research Capital

Quant Trader - Tower Research

Tower Research Capital is a leading quantitative trading firm founded in 1998. Tower has built its business on a high-performance platform and independent trading teams. We have a 25+ year track record of innovation and a reputation for discovering unique market opportunities.

Tower is home to some of the world's best systematic trading and engineering talent. We empower portfolio managers to build their teams and strategies independently while providing the economies of scale that come from a large, global organization.

Engineers thrive at Tower while developing electronic trading infrastructure at a world class level. Our engineers solve challenging problems in the realms of low-latency programming, FPGA technology, hardware acceleration and machine learning. Our ongoing investment in top engineering talent and technology ensures our platform remains unmatched in terms of functionality, scalability and performance.

At Tower, every employee plays a role in our success. Our Business Support teams are essential to building and maintaining the platform that powers everything we do — combining market access, data, compute, and research infrastructure with risk management, compliance, and a full suite of business services. Our Business Support teams enable our trading and engineering teams to perform at their best.

At Tower, employees will find a stimulating, results-oriented environment where highly intelligent and motivated colleagues inspire each other to reach their greatest potential.

Responsibilities
Designing, developing and implementing efficient code for various components of the team's low latency, high throughput production trading and research systems
Production Monitoring and automation of daily tasks
Developing systems that provide easy, highly efficient access to historical market data and trading simulations
Building risk-management and performance-tracking tools
Staying up to date on state-of-the-art technologies in high performance computing industry
Ability to work independently and with minimal supervision
Qualifications
Minimum of 2 years of demonstrated and on the job software development experience preferably in C++
A bachelor's degree or equivalent in computer science or a related field
Knowledge of Linux
Strong background in C/C++ and Python
Strong troubleshooting and problem-solving abilities
The ability to manage multiple tasks in a fast-paced environment
The willingness to take on tasks both big and small
Excellent communication skills and fluency in English`,

  product_management: `Company: Chime

Chime - Product Manager

About the role
We're hiring a Data Platform Product Manager focused on Data Foundations & Insights to help Chime scale how data is built, accessed, and used across the company. In this role, you'll partner closely with engineering, analytics, and cross-functional teams to translate complex stakeholder needs into durable, scalable data products. You'll act as an embedded product partner within XFN squads, owning data foundations that improve time-to-insight, self-serve capabilities, and data trust as Chime continues to grow.

The base salary offered for this role and level of experience will begin at $207,000.00 and up to $244,000.00. Full-time employees are also eligible for a bonus, competitive equity package, and benefits. The actual base salary offered may be higher, depending on your location, skills, qualifications, and experience.

This role is in-office in San Francisco Monday-Thursday.

In this role, you can expect to
Own product strategy and execution for data foundations and insights supporting core products
Partner closely with data engineering and analytics to build scalable, self-serve data platforms and tooling
Embed with cross-functional squads as the data subject-matter expert, translating stakeholder needs into scalable, reusable data products and data platform solutions
Improve time-to-insight and data cycle time for business, product, and compliance use cases
Drive stakeholder satisfaction by delivering reliable, well-documented, and trusted data products
Communicate effectively with both highly technical and non-technical partners across the organization

To thrive in this role, you have
Product management experience owning platform, data, or analytics products in a complex organization
Prior exposure to fintech, financial services, or highly regulated environments (preferred)
Experience partnering closely with data engineering teams on foundations such as data models, pipelines, or metrics layers
Comfort operating as an embedded PM within cross-functional squads
Strong ability to translate ambiguous stakeholder questions into scalable product solutions
Experience supporting analytics, reporting, or insight-generation use cases for multiple audiences
Ability to communicate clearly across technical and business stakeholders`,

  // general: intentionally omitted — see file header
};
