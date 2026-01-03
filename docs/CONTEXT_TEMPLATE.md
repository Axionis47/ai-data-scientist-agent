# Context Document Template

## Purpose

The context document provides essential information about your dataset to enable accurate analysis and question-answering. This document must be in **Word (.docx) format**.

## Required Headings

Your context document **must** include these exact headings (case-sensitive):

1. **Dataset Overview**
2. **Target Use / Primary Questions**
3. **Data Dictionary**
4. **Known Caveats**

## Minimum Requirements

- **Format**: Microsoft Word (.docx) only
- **Headings**: All 4 required headings must be present (exact spelling and capitalization)
- **Content Length**: Minimum 800 characters total
- **Unit of Observation**: Must clearly state what each row represents
- **Data Dictionary**: Must include at least 10 column entries (or all columns if fewer than 10)

## Template Structure

### Dataset Overview

Provide a high-level description of the dataset:
- What does this dataset contain?
- What is the unit of observation (what does each row represent)?
- Time period covered
- Source of the data
- Any relevant context about how it was collected

**Example**:
> This dataset contains customer transaction records from our e-commerce platform for Q1 2024. Each row represents a single transaction. The data includes 50,000 transactions from 12,000 unique customers across 5 product categories.

### Target Use / Primary Questions

Describe the intended use cases and key questions you want to answer:
- What business questions should this dataset help answer?
- What decisions will be informed by this analysis?
- Who is the primary audience?

**Example**:
> Primary questions:
> - What are the top-performing product categories?
> - Which customer segments have the highest lifetime value?
> - What factors correlate with purchase frequency?
> - Are there seasonal patterns in transaction volume?

### Data Dictionary

List all columns (or at least the 10 most important) with:
- Column name
- Data type
- Description
- Example values or range
- Any special notes (nulls allowed, foreign key, etc.)

**Example**:
| Column Name | Type | Description | Notes |
|-------------|------|-------------|-------|
| transaction_id | string | Unique transaction identifier | Primary key |
| customer_id | string | Unique customer identifier | Foreign key |
| transaction_date | date | Date of transaction | Format: YYYY-MM-DD |
| product_category | string | Category of purchased product | 5 categories |
| amount_usd | decimal | Transaction amount in USD | Always positive |
| payment_method | string | Payment method used | credit, debit, paypal |
| is_first_purchase | boolean | Whether this is customer's first purchase | true/false |
| discount_applied | decimal | Discount amount in USD | 0 if no discount |
| shipping_country | string | Country code for shipping | ISO 2-letter code |
| order_status | string | Current order status | pending, shipped, delivered |

### Known Caveats

Document any limitations, data quality issues, or important notes:
- Missing data patterns
- Known biases
- Data quality issues
- Limitations in scope
- Important assumptions

**Example**:
> - Approximately 5% of transactions have missing customer_id (guest checkouts)
> - Data only includes completed transactions (abandoned carts not included)
> - Refunds are recorded as negative amounts in separate rows
> - International transactions converted to USD at daily exchange rate
> - Product category taxonomy changed in March 2024; earlier data mapped to new categories

## Validation

Your document will be validated for:
1. ✅ File format is .docx
2. ✅ All 4 required headings are present
3. ✅ Total content is at least 800 characters
4. ✅ Content is well-structured and readable

## Sample Document

A valid sample document is provided at `docs/samples/context_template.docx`.

## Tips for Success

- Be specific and detailed in your descriptions
- Include concrete examples
- Document edge cases and exceptions
- Update the context document as you learn more about the data
- Keep the language clear and accessible to non-technical stakeholders

